"""
LLM Client Protocol for Enterprise Integration

This module defines the interface that all LLM clients must implement.
Enterprises can create their own implementations to integrate with their
internal AI APIs (Azure OpenAI, AWS Bedrock, internal APIs, etc.)

Example usage for enterprises:

    from agentic.llm_client import LLMClient, Message, ChatResponse

    class MyEnterpriseLLM(LLMClient):
        async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
            # Call your internal API here
            response = await my_internal_api.complete(messages)
            return ChatResponse(
                content=response.text,
                model="internal-model",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
        
        async def close(self):
            pass

    # Then use it:
    from agentic import MasterAgent
    agent = MasterAgent(repo_path, llm_client=MyEnterpriseLLM())
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from dataclasses import dataclass, field


@dataclass
class Message:
    """A chat message."""
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add another TokenUsage to this one."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )
    
    def format(self, context_limit: int = 200000) -> str:
        """Format as human-readable string with context usage."""
        def fmt_tokens(n: int) -> str:
            if n >= 1000:
                return f"{n/1000:.1f}K"
            return str(n)
        
        return f"{fmt_tokens(self.total_tokens)}/{fmt_tokens(context_limit)} tokens"
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TokenUsage':
        """Create from API response usage dict."""
        return cls(
            prompt_tokens=data.get('prompt_tokens', 0),
            completion_tokens=data.get('completion_tokens', 0),
            total_tokens=data.get('total_tokens', 0),
        )


@dataclass
class ChatResponse:
    """Response from a chat completion."""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: Optional[str] = None
    
    @property
    def token_usage(self) -> TokenUsage:
        """Get token usage as TokenUsage object."""
        return TokenUsage.from_dict(self.usage)


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    Implement this interface to integrate with your enterprise AI API.
    
    Required methods:
    - chat(): Send messages and get a response (non-streaming)
    - close(): Clean up resources
    
    Optional methods:
    - chat_stream(): Streaming responses (defaults to non-streaming fallback)
    
    Example Implementation:
    
        class AzureOpenAIClient(LLMClient):
            def __init__(self, endpoint: str, api_key: str, deployment: str):
                self.endpoint = endpoint
                self.api_key = api_key
                self.deployment = deployment
                self._client = None
            
            async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
                # Your Azure OpenAI implementation here
                import openai
                client = openai.AsyncAzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                )
                response = await client.chat.completions.create(
                    model=self.deployment,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    max_tokens=kwargs.get("max_tokens", 4096),
                )
                return ChatResponse(
                    content=response.choices[0].message.content,
                    model=self.deployment,
                    usage=response.usage.model_dump() if response.usage else {},
                )
            
            async def close(self):
                pass
    """
    
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat completion request (non-streaming).
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters like:
                - max_tokens: Maximum tokens in response (default: 4096)
                - temperature: Sampling temperature (default: 0.7)
                - model: Model override (optional)
                
        Returns:
            ChatResponse with the completion content and metadata
        """
        pass
    
    async def chat_stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Send a streaming chat completion request.
        
        Default implementation falls back to non-streaming.
        Override for true streaming support.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Yields:
            Content chunks as they arrive
        """
        # Default: fall back to non-streaming
        response = await self.chat(messages, **kwargs)
        yield response.content
    
    @abstractmethod
    async def close(self):
        """
        Clean up resources (close HTTP clients, etc.)
        
        Called when the agent is done.
        """
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# =============================================================================
# Example implementations for reference
# =============================================================================

class SimpleLLMClient(LLMClient):
    """
    A simple example LLM client that wraps a callable.
    
    Useful for quick testing or simple integrations.
    
    Example:
        async def my_llm(messages):
            # Your logic here
            return "Response text"
        
        client = SimpleLLMClient(my_llm)
    """
    
    def __init__(self, chat_fn):
        """
        Args:
            chat_fn: An async callable that takes list[Message] and returns str
        """
        self.chat_fn = chat_fn
    
    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        result = await self.chat_fn(messages)
        if isinstance(result, ChatResponse):
            return result
        return ChatResponse(
            content=str(result),
            model="custom",
            usage={},
        )
    
    async def close(self):
        pass

