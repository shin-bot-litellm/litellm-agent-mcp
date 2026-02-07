#!/usr/bin/env python3
"""
LiteLLM Agent MCP Server

A simple MCP server that gives AI agents access to 100+ LLMs through LiteLLM.
Designed for agents who want to call different models for different tasks.

Tools:
- call: Call any LLM model
- compare: Compare outputs from multiple models
- models: List available models
- recommend: Get model recommendation for a task type
"""

import asyncio
import json
import os
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Initialize MCP server
server = Server("litellm-agent")

# Model recommendations by task type
MODEL_RECOMMENDATIONS = {
    "code": {
        "model": "gpt-4o",
        "reason": "Strong at code generation, debugging, and review"
    },
    "writing": {
        "model": "claude-sonnet-4-20250514",
        "reason": "Excellent at prose, creative writing, and nuanced text"
    },
    "reasoning": {
        "model": "o1-preview",
        "reason": "Best for complex multi-step reasoning and math"
    },
    "fast": {
        "model": "gpt-4o-mini",
        "reason": "Quick and cheap for simple tasks"
    },
    "long_context": {
        "model": "gemini/gemini-1.5-pro",
        "reason": "Handles very long documents (1M+ tokens)"
    },
    "vision": {
        "model": "gpt-4o",
        "reason": "Strong vision capabilities for image analysis"
    },
    "general": {
        "model": "gpt-4o",
        "reason": "Good all-around model for general tasks"
    }
}

# Common models list
COMMON_MODELS = [
    {"id": "gpt-4o", "provider": "openai", "strengths": ["code", "vision", "general"]},
    {"id": "gpt-4o-mini", "provider": "openai", "strengths": ["fast", "cheap"]},
    {"id": "o1-preview", "provider": "openai", "strengths": ["reasoning", "math"]},
    {"id": "o1-mini", "provider": "openai", "strengths": ["reasoning", "fast"]},
    {"id": "claude-sonnet-4-20250514", "provider": "anthropic", "strengths": ["writing", "analysis"]},
    {"id": "claude-opus-4-20250514", "provider": "anthropic", "strengths": ["complex tasks", "writing"]},
    {"id": "gemini/gemini-1.5-pro", "provider": "google", "strengths": ["long_context", "multimodal"]},
    {"id": "gemini/gemini-1.5-flash", "provider": "google", "strengths": ["fast", "cheap"]},
    {"id": "mistral/mistral-large-latest", "provider": "mistral", "strengths": ["multilingual", "code"]},
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="call",
            description="Call any LLM model through LiteLLM. Use this to get a response from a specific model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model ID (e.g., 'gpt-4o', 'claude-sonnet-4-20250514', 'gemini/gemini-1.5-pro')"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to the model"
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system message"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature (0-2, default 0.7)"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens in response"
                    }
                },
                "required": ["model", "prompt"]
            }
        ),
        Tool(
            name="compare",
            description="Compare responses from multiple models for the same prompt. Useful for getting different perspectives or finding the best answer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of model IDs to compare (e.g., ['gpt-4o', 'claude-sonnet-4-20250514'])"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to all models"
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system message"
                    }
                },
                "required": ["models", "prompt"]
            }
        ),
        Tool(
            name="models",
            description="List available LLM models and their strengths. Use this to see what models you can call.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Filter by provider (openai, anthropic, google, mistral)"
                    }
                }
            }
        ),
        Tool(
            name="recommend",
            description="Get a model recommendation for a specific task type. Helps you pick the right model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "description": "Type of task: code, writing, reasoning, fast, long_context, vision, general"
                    }
                },
                "required": ["task_type"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "call":
        return await handle_call(arguments)
    elif name == "compare":
        return await handle_compare(arguments)
    elif name == "models":
        return await handle_models(arguments)
    elif name == "recommend":
        return await handle_recommend(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_call(args: dict) -> list[TextContent]:
    """Call a single LLM model."""
    try:
        import litellm
        
        model = args["model"]
        prompt = args["prompt"]
        system = args.get("system")
        temperature = args.get("temperature", 0.7)
        max_tokens = args.get("max_tokens")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content
        
        return [TextContent(type="text", text=content)]
        
    except ImportError:
        return [TextContent(type="text", text="Error: litellm not installed. Run: pip install litellm")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error calling {args.get('model', 'unknown')}: {str(e)}")]


async def handle_compare(args: dict) -> list[TextContent]:
    """Compare responses from multiple models."""
    try:
        import litellm
        
        models = args["models"]
        prompt = args["prompt"]
        system = args.get("system")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        results = []
        
        # Call models concurrently
        async def call_model(model: str):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0.7
                )
                return {
                    "model": model,
                    "response": response.choices[0].message.content,
                    "tokens": response.usage.total_tokens if response.usage else None
                }
            except Exception as e:
                return {
                    "model": model,
                    "error": str(e)
                }
        
        tasks = [call_model(m) for m in models]
        results = await asyncio.gather(*tasks)
        
        # Format output
        output = "## Model Comparison\n\n"
        for r in results:
            output += f"### {r['model']}\n"
            if "error" in r:
                output += f"Error: {r['error']}\n\n"
            else:
                output += f"{r['response']}\n"
                if r.get('tokens'):
                    output += f"\n*({r['tokens']} tokens)*\n"
                output += "\n"
        
        return [TextContent(type="text", text=output)]
        
    except ImportError:
        return [TextContent(type="text", text="Error: litellm not installed. Run: pip install litellm")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error comparing models: {str(e)}")]


async def handle_models(args: dict) -> list[TextContent]:
    """List available models."""
    provider_filter = args.get("provider")
    
    models = COMMON_MODELS
    if provider_filter:
        models = [m for m in models if m["provider"] == provider_filter.lower()]
    
    output = "## Available Models\n\n"
    for m in models:
        strengths = ", ".join(m["strengths"])
        output += f"- **{m['id']}** ({m['provider']}): {strengths}\n"
    
    output += "\n*Use `call` with any model ID to get a response.*"
    
    return [TextContent(type="text", text=output)]


async def handle_recommend(args: dict) -> list[TextContent]:
    """Recommend a model for a task type."""
    task_type = args["task_type"].lower()
    
    if task_type not in MODEL_RECOMMENDATIONS:
        available = ", ".join(MODEL_RECOMMENDATIONS.keys())
        return [TextContent(
            type="text", 
            text=f"Unknown task type: {task_type}\n\nAvailable types: {available}"
        )]
    
    rec = MODEL_RECOMMENDATIONS[task_type]
    output = f"## Recommended Model for '{task_type}'\n\n"
    output += f"**Model:** `{rec['model']}`\n"
    output += f"**Reason:** {rec['reason']}\n\n"
    output += f"Use: `call` with model='{rec['model']}'"
    
    return [TextContent(type="text", text=output)]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
