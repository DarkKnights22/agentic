# Enterprise Integration

## Disable Semantic Search

Can't run embedding models? Pick one:

fyi you can also use `python -m agentic.cli`

```bash
# CLI flag
agentic . --no-semantic-search

# Environment variable
export AGENTIC_DISABLE_SEMANTIC_SEARCH=true

# Config file (agentic.config.json)
{"disable_semantic_search": true}
```

Done. The agent will use grep/read_file instead.

---

## Custom LLM API

Don't want OpenRouter? Implement this:

```python
from agentic import LLMClient, Message, ChatResponse, MasterAgent

class MyLLM(LLMClient):
    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse:
        # YOUR API CALL HERE
        result = await your_api(messages)
        return ChatResponse(content=result, model="your-model")
    
    async def close(self):
        pass

# Use it
agent = MasterAgent("/path/to/repo", llm_client=MyLLM())
```

That's it. Two methods: `chat()` and `close()`.

---

## Pre-built Examples

Copy from `agentic/examples/enterprise_llm.py`:

| Class | Use Case |
|-------|----------|
| `AzureOpenAIClient` | Azure OpenAI Service |
| `AWSBedrockClient` | AWS Bedrock (Claude) |
| `SimpleHTTPClient` | Any OpenAI-compatible API |
| `OllamaClient` | Local models via Ollama |

```python
from agentic.examples import AzureOpenAIClient

client = AzureOpenAIClient(
    endpoint="https://your-resource.openai.azure.com",
    api_key="your-key",
    deployment="gpt-4"
)
agent = MasterAgent(repo_path, llm_client=client)
```

---

## Config Reference

`agentic.config.json` in your repo root:

```json
{
  "disable_semantic_search": true,
  "openrouter_model": "anthropic/claude-sonnet-4.5",
  "max_tool_rounds": 20,
  "context_limit": 200000
}
```

Or environment variables:
- `AGENTIC_DISABLE_SEMANTIC_SEARCH=true`
- `AGENTIC_MODEL=your-model`
- `OPENROUTER_API_KEY=your-key`

