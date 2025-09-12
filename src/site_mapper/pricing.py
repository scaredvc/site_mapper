"""
Accurate pricing calculator for OpenAI models
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class PricingInfo:
    """Pricing information for a model"""
    model_name: str
    input_tokens_per_1k: float
    output_tokens_per_1k: float
    image_input_per_1k: float  # For vision models
    supports_vision: bool

# Current OpenAI pricing (as of 2024)
# Source: https://openai.com/pricing
PRICING_DATA = {
    "gpt-3.5-turbo": PricingInfo(
        model_name="gpt-3.5-turbo",
        input_tokens_per_1k=0.0015,
        output_tokens_per_1k=0.002,
        image_input_per_1k=0.0,  # No vision support
        supports_vision=False
    ),
    "gpt-4o-mini": PricingInfo(
        model_name="gpt-4o-mini",
        input_tokens_per_1k=0.00015,
        output_tokens_per_1k=0.0006,
        image_input_per_1k=0.00015,  # Same as input for vision
        supports_vision=True
    ),
    "gpt-4o": PricingInfo(
        model_name="gpt-4o",
        input_tokens_per_1k=0.005,
        output_tokens_per_1k=0.015,
        image_input_per_1k=0.005,  # Same as input for vision
        supports_vision=True
    )
}

class PricingCalculator:
    """Accurate pricing calculator for OpenAI API calls"""
    
    def __init__(self):
        self.pricing_data = PRICING_DATA
    
    def calculate_cost(self, 
                      model_name: str, 
                      input_tokens: int, 
                      output_tokens: int,
                      has_image: bool = False) -> Dict[str, Any]:
        """
        Calculate accurate cost for an API call
        
        Args:
            model_name: Name of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            has_image: Whether the request included an image
            
        Returns:
            Dictionary with cost breakdown
        """
        if model_name not in self.pricing_data:
            # Fallback to gpt-4o-mini pricing for unknown models
            model_name = "gpt-4o-mini"
        
        pricing = self.pricing_data[model_name]
        
        # Calculate text token costs
        input_cost = (input_tokens / 1000) * pricing.input_tokens_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_tokens_per_1k
        
        # Calculate image costs if applicable
        image_cost = 0.0
        if has_image and pricing.supports_vision:
            # For vision models, images add additional cost beyond text tokens
            # This is a simplified calculation - actual image pricing is more complex
            # and depends on image size/resolution
            estimated_image_tokens = 765  # Rough estimate for typical screenshots
            image_cost = (estimated_image_tokens / 1000) * pricing.image_input_per_1k
        
        total_cost = input_cost + output_cost + image_cost
        
        return {
            "total_cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "image_cost": image_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "has_image": has_image,
            "model_name": model_name,
            "pricing_info": {
                "input_rate_per_1k": pricing.input_tokens_per_1k,
                "output_rate_per_1k": pricing.output_tokens_per_1k,
                "image_rate_per_1k": pricing.image_input_per_1k,
                "supports_vision": pricing.supports_vision
            }
        }
    
    def get_model_pricing(self, model_name: str) -> Optional[PricingInfo]:
        """Get pricing information for a specific model"""
        return self.pricing_data.get(model_name)
    
    def list_available_models(self) -> list:
        """Get list of available models with pricing"""
        return list(self.pricing_data.keys())
    
    def update_pricing(self, model_name: str, pricing_info: PricingInfo):
        """Update pricing for a model (useful for price changes)"""
        self.pricing_data[model_name] = pricing_info

# Global instance
pricing_calculator = PricingCalculator()

def calculate_api_cost(model_name: str, 
                      input_tokens: int, 
                      output_tokens: int,
                      has_image: bool = False) -> Dict[str, Any]:
    """Convenience function to calculate API cost"""
    return pricing_calculator.calculate_cost(model_name, input_tokens, output_tokens, has_image)
