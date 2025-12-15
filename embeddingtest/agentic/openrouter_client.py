"""
OpenRouter API Client

Async client for OpenRouter API with streaming support.
Uses Claude Sonnet as the default model for planning tasks.

Implements the LLMClient protocol for enterprise compatibility.
"""

import os
import json
import asyncio
from typing import AsyncGenerator, Optional
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

# Import from the protocol module (with backwards compatibility)
from .llm_client import LLMClient, Message, ChatResponse, TokenUsage

load_dotenv()

# Default model - Change this to use a different OpenRouter model
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# Re-export for backwards compatibility
__all__ = ['OpenRouterClient', 'Message', 'ChatResponse', 'TokenUsage', 'OpenRouterError', 'DEFAULT_MODEL']


class OpenRouterError(Exception):
    """Custom exception for OpenRouter API errors."""
    pass


class OpenRouterClient(LLMClient):
    """
    Async client for OpenRouter API.
    
    Features:
    - Automatic retry with exponential backoff for transient errors (5xx, network)
    - Streaming and non-streaming modes
    
    Usage:
        client = OpenRouterClient()
        response = await client.chat([Message("user", "Hello!")])
        print(response.content)
        
        # Or with streaming:
        async for chunk in client.chat_stream([Message("user", "Hello!")]):
            print(chunk, end="", flush=True)
    """
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 2.0  # seconds, doubles each retry
    RETRYABLE_STATUS_CODES = {500, 502, 503, 504, 520, 521, 522, 523, 524}
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def headers(self) -> dict:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/agentic-cli",
            "X-Title": "Agentic CLI",
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=self.headers,
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _build_payload(
        self,
        messages: list[Message],
        stream: bool = False,
        **kwargs
    ) -> dict:
        """Build the request payload."""
        return {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": stream,
        }
    
    async def chat(
        self,
        messages: list[Message],
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat completion request (non-streaming).
        
        Includes automatic retry with exponential backoff for transient errors.
        
        Args:
            messages: List of Message objects
            **kwargs: Override default parameters (model, max_tokens, temperature)
            
        Returns:
            ChatResponse with the completion
        """
        client = await self._get_client()
        payload = self._build_payload(messages, stream=False, **kwargs)
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(OPENROUTER_API_URL, json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                if "error" in data:
                    raise OpenRouterError(f"API error: {data['error']}")
                
                # Validate response structure
                if "choices" not in data or not data["choices"]:
                    raise OpenRouterError(f"Invalid API response: no choices returned. Response: {data}")
                
                choice = data["choices"][0]
                
                if "message" not in choice or "content" not in choice["message"]:
                    raise OpenRouterError(f"Invalid choice format: {choice}")
                
                return ChatResponse(
                    content=choice["message"]["content"],
                    model=data.get("model", self.model),
                    usage=data.get("usage", {}),
                    finish_reason=choice.get("finish_reason"),
                )
                
            except httpx.HTTPStatusError as e:
                last_error = e
                # Retry on transient server errors
                if e.response.status_code in self.RETRYABLE_STATUS_CODES:
                    if attempt < self.max_retries:
                        delay = self.RETRY_DELAY_BASE * (2 ** attempt)
                        print(f"[Retry] {e.response.status_code} error, waiting {delay:.1f}s... (attempt {attempt + 1}/{self.max_retries + 1})")
                        await asyncio.sleep(delay)
                        continue
                # Non-retryable or max retries reached
                error_detail = e.response.text[:500]  # Truncate HTML garbage
                raise OpenRouterError(f"API request failed: {e.response.status_code} - {error_detail}")
                
            except httpx.RequestError as e:
                last_error = e
                # Retry on network errors
                if attempt < self.max_retries:
                    delay = self.RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"[Retry] Network error, waiting {delay:.1f}s... (attempt {attempt + 1}/{self.max_retries + 1})")
                    await asyncio.sleep(delay)
                    continue
                raise OpenRouterError(f"Network error: {e}")
        
        # Should not reach here, but just in case
        raise OpenRouterError(f"Max retries exceeded. Last error: {last_error}")
    
    async def chat_stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Send a streaming chat completion request.
        
        Args:
            messages: List of Message objects
            **kwargs: Override default parameters
            
        Yields:
            Content chunks as they arrive
        """
        client = await self._get_client()
        payload = self._build_payload(messages, stream=True, **kwargs)
        
        try:
            async with client.stream("POST", OPENROUTER_API_URL, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:]  # Remove "data: " prefix
                    
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    
                    if "error" in data:
                        raise OpenRouterError(f"Stream error: {data['error']}")
                    
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                            
        except httpx.HTTPStatusError as e:
            raise OpenRouterError(f"Stream request failed: {e.response.status_code}")
        except httpx.RequestError as e:
            raise OpenRouterError(f"Network error during stream: {e}")


async def test_client():
    """Quick test of the client."""
    async with OpenRouterClient() as client:
        print("Testing non-streaming...")
        response = await client.chat([
            Message("system", "You are a helpful assistant."),
            Message("user", "Say hello in exactly 5 words."),
        ])
        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        
        print("\nTesting streaming...")
        async for chunk in client.chat_stream([
            Message("system", "You are a helpful assistant."),
            Message("user", "Count from 1 to 5, one number per line."),
        ]):
            print(chunk, end="", flush=True)
        print("\n\nDone!")


if __name__ == "__main__":
    asyncio.run(test_client())

