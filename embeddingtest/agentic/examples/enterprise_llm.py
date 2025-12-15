"""
Enterprise LLM Integration Examples

This file shows how enterprises can integrate their own LLM APIs with Agentic.
Copy and modify these examples for your specific setup.

Quick Start:
1. Create a class that inherits from LLMClient
2. Implement the `chat()` and `close()` methods
3. Pass your client to MasterAgent

    from agentic import MasterAgent
    from my_enterprise_llm import MyCompanyLLM
    
    agent = MasterAgent("/path/to/repo", llm_client=MyCompanyLLM())
"""

from typing import Optional
from agentic.llm_client import LLMClient, Message, ChatResponse


# =============================================================================
# Example 1: Azure OpenAI
# =============================================================================

class AzureOpenAIClient(LLMClient):
    """
    Example integration with Azure OpenAI Service.
    
    Usage:
        client = AzureOpenAIClient(
            endpoint="https://your-resource.openai.azure.com",
            api_key="your-api-key",
            deployment="gpt-4",  # Your deployment name
        )
        agent = MasterAgent(repo_path, llm_client=client)
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-02-01",
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version
        self._client = None
    
    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        # Lazy import - only needed if using Azure
        from openai import AsyncAzureOpenAI
        
        if self._client is None:
            self._client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        
        response = await self._client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
        )
        
        return ChatResponse(
            content=response.choices[0].message.content or "",
            model=self.deployment,
            usage=response.usage.model_dump() if response.usage else {},
            finish_reason=response.choices[0].finish_reason,
        )
    
    async def close(self):
        self._client = None


# =============================================================================
# Example 2: AWS Bedrock (Claude)
# =============================================================================

class AWSBedrockClient(LLMClient):
    """
    Example integration with AWS Bedrock (Claude models).
    
    Usage:
        client = AWSBedrockClient(
            region="us-east-1",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        )
        agent = MasterAgent(repo_path, llm_client=client)
    
    Requires: boto3, configured AWS credentials
    """
    
    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    ):
        self.region = region
        self.model_id = model_id
        self._client = None
    
    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        import boto3
        import json
        import asyncio
        
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        
        # Separate system message from conversation
        system_content = ""
        conversation = []
        
        for msg in messages:
            if msg.role == "system":
                system_content += msg.content + "\n"
            else:
                conversation.append({"role": msg.role, "content": msg.content})
        
        # Build Bedrock request
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": conversation,
        }
        
        if system_content:
            body["system"] = system_content.strip()
        
        # Run synchronous boto3 in thread pool
        def invoke():
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
            )
            return json.loads(response["body"].read())
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, invoke)
        
        return ChatResponse(
            content=result["content"][0]["text"],
            model=self.model_id,
            usage={
                "prompt_tokens": result.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": result.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    result.get("usage", {}).get("input_tokens", 0) +
                    result.get("usage", {}).get("output_tokens", 0)
                ),
            },
        )
    
    async def close(self):
        self._client = None


# =============================================================================
# Example 3: Simple HTTP API
# =============================================================================

class SimpleHTTPClient(LLMClient):
    """
    Example integration with a simple HTTP-based LLM API.
    
    This works with any API that accepts:
    POST /chat/completions
    {
        "messages": [{"role": "...", "content": "..."}],
        "max_tokens": 4096
    }
    
    And returns:
    {
        "content": "response text",
        "model": "model-name"
    }
    
    Usage:
        client = SimpleHTTPClient(
            base_url="https://internal-llm.yourcompany.com",
            api_key="your-internal-key",
        )
        agent = MasterAgent(repo_path, llm_client=client)
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.custom_headers = headers or {}
        self._client = None
    
    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        import httpx
        
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            headers.update(self.custom_headers)
            
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=120.0,
            )
        
        payload = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        
        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        # Handle OpenAI-style response
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
        else:
            content = data.get("content", data.get("response", ""))
        
        return ChatResponse(
            content=content,
            model=data.get("model", "unknown"),
            usage=data.get("usage", {}),
        )
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Example 4: Function Wrapper (Quick and Easy)
# =============================================================================

from agentic.llm_client import SimpleLLMClient


def create_function_client(sync_fn):
    """
    Wrap a simple function as an LLM client.
    
    The function should accept a list of Message objects and return a string.
    
    Usage:
        def my_llm(messages):
            # Your logic here
            prompt = "\\n".join(f"{m.role}: {m.content}" for m in messages)
            return call_my_internal_api(prompt)
        
        client = create_function_client(my_llm)
        agent = MasterAgent(repo_path, llm_client=client)
    """
    import asyncio
    
    async def async_wrapper(messages):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, sync_fn, messages)
        return result
    
    return SimpleLLMClient(async_wrapper)


# =============================================================================
# Example 5: Ollama (Local Models)
# =============================================================================

class OllamaClient(LLMClient):
    """
    Example integration with Ollama for local model inference.
    
    Usage:
        client = OllamaClient(model="llama3")
        agent = MasterAgent(repo_path, llm_client=client)
    
    Requires: Ollama running locally (ollama serve)
    """
    
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = None
    
    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        import httpx
        
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=300.0)
        
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        
        response = await self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        return ChatResponse(
            content=data["message"]["content"],
            model=self.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
        )
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

