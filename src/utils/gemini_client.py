"""Gemini API client wrapper with error handling"""

import os
import time
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper for Gemini API with error handling and retry logic"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-pro"):
        """Initialize Gemini client
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model name to use
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key must be provided or set in GEMINI_API_KEY environment variable")
        
        self.model = model
        self.max_retries = 3
        self.retry_delay = 2.0
        
        # Import google.generativeai here to avoid import errors if not installed
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.model_instance = genai.GenerativeModel(model)
        except ImportError:
            logger.warning("google-generativeai not installed. Client will not be functional.")
            self.genai = None
            self.model_instance = None
    
    def generate_content(self, prompt: str, **kwargs) -> Any:
        """Generate content with retry logic
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
            
        Raises:
            Exception: If all retries fail
        """
        if not self.model_instance:
            raise RuntimeError("Gemini client not properly initialized")
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.model_instance.generate_content(prompt, **kwargs)
                return response
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise Exception(f"Gemini API call failed after {self.max_retries} attempts: {last_error}")
    
    def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON response
        
        Args:
            prompt: Input prompt (should request JSON format)
            **kwargs: Additional generation parameters
            
        Returns:
            Parsed JSON response
        """
        import json
        
        response = self.generate_content(prompt, **kwargs)
        try:
            # Extract JSON from response text
            text = response.text
            # Try to find JSON in the response
            if "```json" in text:
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                text = text[json_start:json_end].strip()
            elif "```" in text:
                json_start = text.find("```") + 3
                json_end = text.find("```", json_start)
                text = text[json_start:json_end].strip()
            
            return json.loads(text)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse JSON from Gemini response: {e}")
            logger.error(f"Response text: {response.text if hasattr(response, 'text') else response}")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")
