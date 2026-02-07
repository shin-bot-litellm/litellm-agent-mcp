# LiteLLM Agent MCP Server

**Give your AI agent access to 100+ LLMs.**

This MCP server lets AI agents (Claude Code, Cursor, etc.) call any LLM through LiteLLM's unified API. Stop being limited to one model — use the right model for each task.

## Why?

AI agents are typically stuck on a single model. With this MCP server, your agent can:

- 🔀 **Call any model** — GPT-4, Claude, Gemini, Mistral, and 100+ more
- ⚖️ **Compare outputs** — Get responses from multiple models and pick the best
- 🎯 **Use the right tool** — Code tasks → GPT-4, writing → Claude, long docs → Gemini
- 💰 **Save costs** — Route simple queries to cheaper models

## Tools

| Tool | Description |
|------|-------------|
| `call` | Call any LLM model with a prompt |
| `compare` | Compare responses from multiple models |
| `models` | List available models and their strengths |
| `recommend` | Get model recommendation for a task type |

## Installation

### Claude Desktop / Cursor

Add to your MCP config:

```json
{
  "mcpServers": {
    "litellm": {
      "command": "python",
      "args": ["-m", "litellm_agent_mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-..."
      }
    }
  }
}
```

### From PyPI

```bash
pip install litellm-agent-mcp
```

### From Source

```bash
git clone https://github.com/shin-bot-litellm/litellm-agent-mcp
cd litellm-agent-mcp
pip install -e .
```

## Usage Examples

### Call a specific model

```
Use the `call` tool:
- model: "gpt-4o"  
- prompt: "Explain this code: [code here]"
```

### Compare multiple models

```
Use the `compare` tool:
- models: ["gpt-4o", "claude-sonnet-4-20250514"]
- prompt: "What's the best approach to implement caching?"
```

### Get a recommendation

```
Use the `recommend` tool:
- task_type: "code"

→ Returns: gpt-4o (Strong at code generation, debugging, and review)
```

## Environment Variables

Set API keys for the providers you want to use:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
```

Or point to a LiteLLM proxy:

```bash
LITELLM_API_BASE=https://your-proxy.com
LITELLM_API_KEY=sk-...
```

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, o1-preview, o1-mini |
| Anthropic | claude-sonnet-4, claude-opus-4 |
| Google | gemini-1.5-pro, gemini-1.5-flash |
| Mistral | mistral-large-latest |
| + 100 more | See [LiteLLM docs](https://docs.litellm.ai/docs/providers) |

## License

MIT

## Links

- [LiteLLM Docs](https://docs.litellm.ai)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [MCP Spec](https://modelcontextprotocol.io)
