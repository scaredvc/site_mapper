"""
AI Classifier Module for Link Classification Bake-off

This module contains functions to classify links as either 'content_link' or 'filter_link'
using two different AI models: OpenAI's GPT-4o and Hugging Face's Sightseer.
"""

import base64
import json
import os
import time
from typing import Dict, Any, Optional
from io import BytesIO

import requests
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()


class AIClassifier:
    """Base class for AI-based link classification"""
    
    def __init__(self):
        self.model_name = "unknown"
        self.total_cost = 0.0
        self.total_time = 0.0
        self.request_count = 0
    
    def classify_link(self, link_data: Dict[str, Any], screenshot_path: str) -> Dict[str, Any]:
        """
        Classify a link as content_link or filter_link
        
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
            "avg_time_per_request": self.total_time / max(self.request_count, 1)
        }


class GPT4oClassifier(AIClassifier):
    """Classifier using OpenAI's GPT-3.5-turbo model"""
    
    def __init__(self):
        super().__init__()
        self.model_name = "GPT-3.5-turbo"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
    
    def _prepare_image(self, screenshot_path: str) -> str:
        """Convert image to base64 for OpenAI API"""
        try:
            with open(screenshot_path, "rb") as image_file:
                # Resize image if too large (OpenAI has size limits)
                image = Image.open(image_file)
                # Resize to max 2048x2048 while maintaining aspect ratio
                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                return image_base64
        except Exception as e:
            raise Exception(f"Error processing image: {e}")
    
    def _create_prompt(self, link_data: Dict[str, Any]) -> str:
        """Create a detailed prompt for GPT-4o"""
        analysis = link_data.get('analysis', {})
        
        prompt = f"""
You are an expert web analyst tasked with classifying links on archive-it.org. Your job is to determine if a link is a "content_link" or "filter_link".

CONTENT_LINK examples:
- Individual Collection Pages: Links with collection titles like "Web Archiving Service for United States Government Information"
- Organization/Creator Pages: Links to content creator profiles like "U.S. Government Publishing Office"

FILTER_LINK examples:
- Facet Filters: Links under "FILTER BY" sidebar (Subject, Creator categories)
- Sorting Controls: Links for ordering like "Relevance", "Title", "Date Created"
- Pagination: Page number links (1, 2, 3, Next)
- View Controls: Buttons for list/grid view switching

Link Data:
- Text: "{link_data.get('text', '')}"
- URL: {link_data.get('absolute_url', '')}
- DOM Hierarchy: {analysis.get('dom_hierarchy', 'N/A')}
- CSS Classes: {analysis.get('css_classes', [])}
- Link Position: {analysis.get('link_position', 'N/A')}
- Parent Elements: {analysis.get('parent_elements', [])}
- Bounding Box: {analysis.get('bounding_box', {})}

Look at the screenshot and the link data above. Classify this link as either "content_link" or "filter_link".

Respond with ONLY the classification (content_link or filter_link), no explanation.
"""
        return prompt.strip()
    
    def classify_link(self, link_data: Dict[str, Any], screenshot_path: str) -> Dict[str, Any]:
        """Classify link using GPT-4o"""
        start_time = time.time()
        
        try:
            # Prepare the image
            image_base64 = self._prepare_image(screenshot_path)
            
            # Create the prompt
            prompt = self._create_prompt(link_data)
            
            # Make API call (GPT-3.5-turbo text only, no image)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            # Extract classification
            classification = response.choices[0].message.content.strip().lower()
            if classification not in ["content_link", "filter_link"]:
                classification = "unknown"
            
            # Calculate cost (GPT-3.5-turbo pricing)
            # Input: ~$0.0015 per 1K tokens, Output: ~$0.002 per 1K tokens
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * 0.0015 / 1000) + (output_tokens * 0.002 / 1000)
            
            elapsed_time = time.time() - start_time
            
            # Update stats
            self.request_count += 1
            self.total_cost += cost
            self.total_time += elapsed_time
            
            return {
                "classification": classification,
                "confidence": 1.0,  # GPT-4o doesn't provide confidence scores
                "model": self.model_name,
                "processing_time": elapsed_time,
                "cost": cost,
                "tokens_used": input_tokens + output_tokens,
                "success": True
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.request_count += 1
            self.total_time += elapsed_time
            
            return {
                "classification": "error",
                "confidence": 0.0,
                "model": self.model_name,
                "processing_time": elapsed_time,
                "cost": 0.0,
                "error": str(e),
                "success": False
            }


class SightseerClassifier(AIClassifier):
    """Classifier using Hugging Face Sightseer model"""
    
    def __init__(self):
        super().__init__()
        self.model_name = "Sightseer"
        self.api_url = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"
        self.api_token = os.getenv("HUGGING_FACE_API_TOKEN")
        if not self.api_token:
            raise ValueError("HUGGING_FACE_API_TOKEN not found in environment variables")
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
    
    def _prepare_image(self, screenshot_path: str) -> str:
        """Convert image to base64 for Hugging Face API"""
        try:
            with open(screenshot_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode()
                return image_base64
        except Exception as e:
            raise Exception(f"Error processing image: {e}")
    
    def _create_prompt(self, link_data: Dict[str, Any]) -> str:
        """Create a detailed prompt for Sightseer"""
        analysis = link_data.get('analysis', {})
        
        prompt = f"""
<image>
Link Classification Task:

Classify this link as either "content_link" or "filter_link".

CONTENT_LINK examples:
- Collection pages with titles like "Web Archiving Service for United States Government Information"
- Organization/Creator profile pages

FILTER_LINK examples:
- Facet filters under "FILTER BY" sidebar
- Sorting controls like "Relevance", "Title", "Date Created"
- Pagination links (1, 2, 3, Next)
- View control buttons

Link Data:
Text: "{link_data.get('text', '')}"
URL: {link_data.get('absolute_url', '')}
DOM: {analysis.get('dom_hierarchy', 'N/A')}
Position: {analysis.get('link_position', 'N/A')}
Classes: {analysis.get('css_classes', [])}

Respond with only: content_link or filter_link
"""
        return prompt.strip()
    
    def classify_link(self, link_data: Dict[str, Any], screenshot_path: str) -> Dict[str, Any]:
        """Classify link using Sightseer"""
        start_time = time.time()
        
        try:
            # Prepare the image
            image_base64 = self._prepare_image(screenshot_path)
            
            # Create the prompt
            prompt = self._create_prompt(link_data)
            
            # Prepare the payload
            payload = {
                "inputs": f"{prompt}<image>{image_base64}",
                "parameters": {
                    "max_new_tokens": 10,
                    "temperature": 0.1,
                    "return_full_text": False
                }
            }
            
            # Make API call
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            result = response.json()
            
            # Extract classification from response
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "").strip().lower()
                # Clean up the response
                if "content_link" in generated_text:
                    classification = "content_link"
                elif "filter_link" in generated_text:
                    classification = "filter_link"
                else:
                    classification = "unknown"
            else:
                classification = "unknown"
            
            elapsed_time = time.time() - start_time
            
            # Update stats (Hugging Face Inference API is free for most models)
            self.request_count += 1
            self.total_cost += 0.0  # Free tier
            self.total_time += elapsed_time
            
            return {
                "classification": classification,
                "confidence": 1.0,  # Sightseer doesn't provide confidence scores
                "model": self.model_name,
                "processing_time": elapsed_time,
                "cost": 0.0,
                "raw_response": result,
                "success": True
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.request_count += 1
            self.total_time += elapsed_time
            
            return {
                "classification": "error",
                "confidence": 0.0,
                "model": self.model_name,
                "processing_time": elapsed_time,
                "cost": 0.0,
                "error": str(e),
                "success": False
            }


def create_classifier(model_name: str) -> AIClassifier:
    """Factory function to create classifier instances"""
    if model_name.lower() == "gpt4o":
        return GPT4oClassifier()
    elif model_name.lower() == "sightseer":
        return SightseerClassifier()
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: gpt4o, sightseer")


# Convenience functions for direct use
def classify_with_gpt4o(link_data: Dict[str, Any], screenshot_path: str) -> Dict[str, Any]:
    """Classify a link using GPT-4o"""
    classifier = GPT4oClassifier()
    return classifier.classify_link(link_data, screenshot_path)


def classify_with_sightseer(link_data: Dict[str, Any], screenshot_path: str) -> Dict[str, Any]:
    """Classify a link using Sightseer"""
    classifier = SightseerClassifier()
    return classifier.classify_link(link_data, screenshot_path)
