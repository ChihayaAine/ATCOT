"""
LLM interface utilities for the ATCOT framework.

This module provides abstractions for interacting with various language models
including OpenAI GPT, Anthropic Claude, and local models.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    tokens_used: int = 0
    finish_reason: str = "completed"
    model: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7,
                max_tokens: int = 512,
                **kwargs) -> str:
        """Generate text synchronously."""
        pass

    @abstractmethod
    async def generate_async(self, 
                           prompt: str, 
                           temperature: float = 0.7,
                           max_tokens: int = 512,
                           **kwargs) -> str:
        """Generate text asynchronously."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "interface_type": self.__class__.__name__
        }


class OpenAIInterface(LLMInterface):
    """Interface for OpenAI GPT models."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client

    def generate(self, 
                prompt: str, 
                temperature: float = 0.7,
                max_tokens: int = 512,
                **kwargs) -> str:
        """Generate text using OpenAI API."""
        try:
            client = self._get_client()
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return f"[Error: {str(e)}]"

    async def generate_async(self, 
                           prompt: str, 
                           temperature: float = 0.7,
                           max_tokens: int = 512,
                           **kwargs) -> str:
        """Generate text asynchronously using OpenAI API."""
        try:
            # Use asyncio to run the sync method in a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self.generate(prompt, temperature, max_tokens, **kwargs)
            )
        except Exception as e:
            logger.error(f"OpenAI async generation failed: {e}")
            return f"[Error: {str(e)}]"


class AnthropicInterface(LLMInterface):
    """Interface for Anthropic Claude models."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self._client = None

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        return self._client

    def generate(self, 
                prompt: str, 
                temperature: float = 0.7,
                max_tokens: int = 512,
                **kwargs) -> str:
        """Generate text using Anthropic API."""
        try:
            client = self._get_client()
            
            response = client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            return f"[Error: {str(e)}]"

    async def generate_async(self, 
                           prompt: str, 
                           temperature: float = 0.7,
                           max_tokens: int = 512,
                           **kwargs) -> str:
        """Generate text asynchronously using Anthropic API."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self.generate(prompt, temperature, max_tokens, **kwargs)
            )
        except Exception as e:
            logger.error(f"Anthropic async generation failed: {e}")
            return f"[Error: {str(e)}]"


class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing and demonstration."""
    
    def __init__(self, model_name: str = "mock-llm"):
        super().__init__(model_name)

    def generate(self, 
                prompt: str, 
                temperature: float = 0.7,
                max_tokens: int = 512,
                **kwargs) -> str:
        """Generate mock response."""
        # Simple mock responses based on prompt content
        prompt_lower = prompt.lower()
        
        if "plan" in prompt_lower and "step" in prompt_lower:
            return self._generate_plan_response()
        elif "confidence" in prompt_lower:
            return "0.8"
        elif "contradiction" in prompt_lower:
            return "No contradiction detected."
        elif "tool" in prompt_lower and "select" in prompt_lower:
            return "0.7"
        elif "final answer" in prompt_lower:
            return self._generate_final_answer_response()
        elif "revise" in prompt_lower or "correction" in prompt_lower:
            return self._generate_revision_response()
        else:
            return f"Mock response to: {prompt[:50]}..."

    async def generate_async(self, 
                           prompt: str, 
                           temperature: float = 0.7,
                           max_tokens: int = 512,
                           **kwargs) -> str:
        """Generate mock response asynchronously."""
        # Simulate async processing
        await asyncio.sleep(0.1)
        return self.generate(prompt, temperature, max_tokens, **kwargs)

    def _generate_plan_response(self) -> str:
        """Generate mock planning response."""
        return """Step 1: Analyze the query and identify key information needed
Dependencies: None
Tools: None
Confidence: 0.9

Step 2: Gather necessary information using appropriate tools
Dependencies: 1
Tools: web_search, calculator
Confidence: 0.8

Step 3: Process and integrate the gathered information
Dependencies: 2
Tools: python_interpreter
Confidence: 0.7

Step 4: Formulate final answer based on analysis
Dependencies: 3
Tools: None
Confidence: 0.8"""

    def _generate_final_answer_response(self) -> str:
        """Generate mock final answer response."""
        return """Based on the reasoning process and available information, the answer is:

Final Answer: The mock system has successfully processed the query using the ATCOT framework, demonstrating adaptive planning, tool execution, and correction mechanisms."""

    def _generate_revision_response(self) -> str:
        """Generate mock revision response."""
        return """Step: Incorporating new information from tool observation
Justification: The tool result provides updated information that resolves the previous contradiction
Confidence: 0.8

Step: Adjusting reasoning based on corrected information
Justification: Logical flow is maintained while incorporating the new data
Confidence: 0.9"""


class LLMInterfaceFactory:
    """Factory for creating LLM interfaces."""
    
    @staticmethod
    def create_interface(provider: str, 
                        model_name: Optional[str] = None,
                        api_key: Optional[str] = None,
                        **kwargs) -> LLMInterface:
        """Create an LLM interface based on provider."""
        provider_lower = provider.lower()
        
        if provider_lower == "openai":
            model_name = model_name or "gpt-4"
            return OpenAIInterface(model_name, api_key)
        
        elif provider_lower == "anthropic":
            model_name = model_name or "claude-3-sonnet-20240229"
            return AnthropicInterface(model_name, api_key)
        
        elif provider_lower == "mock":
            model_name = model_name or "mock-llm"
            return MockLLMInterface(model_name)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available LLM providers."""
        return ["openai", "anthropic", "mock"]
