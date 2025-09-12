"""
Generalized AI Classifier for Link Classification

This module provides a generalized classifier that works with multiple AI models
for testing link classification across different websites.
"""

import base64
import json
import os
import time
from typing import Dict, Any, Optional, List
from io import BytesIO
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

from .pricing import calculate_api_cost

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for an AI model"""
    name: str
    api_key_env: str
    model_name: str
    supports_vision: bool
    cost_per_1k_input: float
    cost_per_1k_output: float


class GeneralizedClassifier:
    """Base class for generalized AI-based link classification"""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.model_name = model_config.name
        self.total_cost = 0.0
        self.total_time = 0.0
        self.request_count = 0
        self.classifications = []
    
    def classify_link(self, link_data: Dict[str, Any], screenshot_path: str) -> Dict[str, Any]:
        """
        Classify a link as content_link, filter_link, or navigation_link
        
        Args:
            link_data: Dictionary containing link analysis data
            screenshot_path: Path to the screenshot image
            
        Returns:
            Dictionary with classification result and metadata
        """
        raise NotImplementedError("Subclasses must implement classify_link method")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this classifier"""
        return {
            "model_name": self.model_name,
            "total_requests": self.request_count,
            "total_cost": self.total_cost,
            "total_time": self.total_time,
            "avg_time_per_request": self.total_time / max(self.request_count, 1),
            "classifications": self.classifications
        }


class OpenAIClassifier(GeneralizedClassifier):
    """Classifier using OpenAI models (GPT-3.5, GPT-4, GPT-4o)"""
    
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        api_key = os.getenv(model_config.api_key_env)
        if not api_key:
            raise ValueError(f"{model_config.api_key_env} not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        # Simple rate limiting (requests per minute). Override with CLASSIFIER_RPM env.
        try:
            rpm_env = os.getenv("CLASSIFIER_RPM")
            rpm = int(rpm_env) if rpm_env else 60
            if rpm <= 0:
                rpm = 1
        except Exception:
            rpm = 60
        self._rpm = rpm
        self._min_interval = 60.0 / float(self._rpm)
        self._last_request_time: Optional[float] = None
    
    def _prepare_image(self, screenshot_path: str) -> str:
        """Convert image to base64 for OpenAI API"""
        try:
            with open(screenshot_path, "rb") as image_file:
                image = Image.open(image_file)
                # Resize to max 1024x1024 for better performance
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                buffer = BytesIO()
                image.save(buffer, format="PNG", optimize=True)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                return image_base64
        except Exception as e:
            raise Exception(f"Error processing image: {e}")
    
    def _create_prompt(self, link_data: Dict[str, Any], domain: str) -> str:
        """Create a generalized prompt for any website"""
        analysis = link_data.get('analysis', {})
        
        # Create ultra-concise prompts that force single-word responses
        import random
        prompt_variations = [
            f"""Link: "{link_data.get('text', '')}"
URL: {link_data.get('absolute_url', '')}
Context: {analysis.get('dom_hierarchy', 'N/A')}

Answer with ONE word: content_link, filter_link, or navigation_link""",
            
            f"""Text: "{link_data.get('text', '')}"
URL: {link_data.get('absolute_url', '')}
DOM: {analysis.get('dom_hierarchy', 'N/A')}

Classify: content_link, filter_link, navigation_link""",
            
            f"""Link: "{link_data.get('text', '')}"
URL: {link_data.get('absolute_url', '')}
Context: {analysis.get('dom_hierarchy', 'N/A')}

Type: content_link, filter_link, navigation_link"""
        ]
        
        prompt = random.choice(prompt_variations)
        return prompt.strip()
    
    def classify_link(self, link_data: Dict[str, Any], screenshot_path: str) -> Dict[str, Any]:
        """Classify link using OpenAI model"""
        start_time = time.time()
        
        try:
            # Extract domain from URL
            url = link_data.get('absolute_url', '')
            domain = url.split('/')[2] if '://' in url else 'this website'
            
            # Create the prompt
            prompt = self._create_prompt(link_data, domain)
            
            # Prepare image if screenshot exists
            image_base64 = None
            if screenshot_path and os.path.exists(screenshot_path):
                try:
                    image_base64 = self._prepare_image(screenshot_path)
                except Exception as e:
                    print(f"Warning: Could not process image {screenshot_path}: {e}")
            
            # Prepare messages with system instruction
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a link classifier. Respond with EXACTLY ONE UPPERCASE LETTER and NOTHING ELSE: "
                        "C = content_link, F = filter_link, N = navigation_link. "
                        "Do not include punctuation, spaces, or words. Output must be a single character from {C,F,N}."
                    )
                }
            ]
            
            # Add user message
            user_content = [{"type": "text", "text": prompt}]
            if image_base64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                })
            
            messages.append({"role": "user", "content": user_content})
            
            # Enforce simple rate limit
            now = time.time()
            if self._last_request_time is not None:
                since_last = now - self._last_request_time
                if since_last < self._min_interval:
                    time.sleep(self._min_interval - since_last)

            # Make API call with retries and timeout
            max_retries = 3
            backoff = 0.5
            last_exc: Optional[Exception] = None
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        max_tokens=1,
                        temperature=0.1,
                        timeout=30,
                    )
                    break
                except Exception as e:
                    last_exc = e
                    # Basic detection of transient/rate-limit errors
                    msg = str(e).lower()
                    if attempt < max_retries - 1 and ("rate" in msg or "timeout" in msg or "temporar" in msg or "429" in msg or "503" in msg):
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    else:
                        raise

            self._last_request_time = time.time()
            
            # Extract single-letter classification and map to full label
            raw = (response.choices[0].message.content or "").strip()
            # Take only the first character and normalize
            raw_letter = (raw[0] if raw else "").lower()
            letter_map = {"c": "content_link", "f": "filter_link", "n": "navigation_link"}
            classification = letter_map.get(raw_letter, "unknown")
            letter_labels = ["c", "f", "n"]
            
            # Calculate confidence and log probability from logprobs
            # Confidence/logprobs are not computed/reported
            
            # Calculate accurate cost using pricing calculator
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            has_image = image_base64 is not None
            
            
            cost_info = calculate_api_cost(
                model_name=self.config.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                has_image=has_image
            )
            cost = cost_info["total_cost"]
            
            elapsed_time = time.time() - start_time
            
            # Update stats
            self.request_count += 1
            self.total_cost += cost
            self.total_time += elapsed_time
            self.classifications.append(classification)
            
            return {
                "classification": classification,
                "model": self.model_name,
                "processing_time": elapsed_time,
                "cost": cost,
                "tokens_used": input_tokens + output_tokens,
                "cost_breakdown": cost_info,
                "success": True
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.request_count += 1
            self.total_time += elapsed_time
            
            return {
                "classification": "error",
                "model": self.model_name,
                "processing_time": elapsed_time,
                "cost": 0.0,
                "error": str(e),
                "success": False
            }


# Available model configurations
AVAILABLE_MODELS = {
    "gpt-3.5-turbo": ModelConfig(
        name="GPT-3.5 Turbo",
        api_key_env="OPENAI_API_KEY",
        model_name="gpt-3.5-turbo",
        supports_vision=False,
        cost_per_1k_input=0.0015,
        cost_per_1k_output=0.002
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        api_key_env="OPENAI_API_KEY",
        model_name="gpt-4o-mini",
        supports_vision=True,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        api_key_env="OPENAI_API_KEY",
        model_name="gpt-4o",
        supports_vision=True,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015
    )
}


def create_classifier(model_id: str) -> GeneralizedClassifier:
    """Factory function to create classifier instances"""
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    config = AVAILABLE_MODELS[model_id]
    return OpenAIClassifier(config)


def get_available_models() -> List[str]:
    """Get list of available model IDs"""
    return list(AVAILABLE_MODELS.keys())


def check_api_keys() -> Dict[str, bool]:
    """Check which models have API keys available"""
    status = {}
    for model_id, config in AVAILABLE_MODELS.items():
        api_key = os.getenv(config.api_key_env)
        status[model_id] = api_key is not None and len(api_key.strip()) > 0
    return status
