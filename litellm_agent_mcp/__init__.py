#!/usr/bin/env python3
"""
LiteLLM Agent MCP Server

A simple MCP server that gives AI agents access to 100+ LLMs through LiteLLM.
Designed for agents who want to call different models for different tasks.

Tools:
- call: Call any LLM model (OpenAI chat completions format)
- responses: Use OpenAI Responses API format
- messages: Use Anthropic Messages API format
- generate_content: Use Google generateContent format
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
            description="Call any LLM model through LiteLLM using OpenAI chat completions format. This is the standard way to get a response from a model.",
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
            name="responses",
            description="Use OpenAI Responses API format. Supports stateful conversations, built-in tools, and structured outputs. Best for complex multi-turn interactions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model ID (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')"
                    },
                    "input": {
                        "type": "string",
                        "description": "The input text/prompt"
                    },
                    "instructions": {
                        "type": "string",
                        "description": "System instructions for the model"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature (0-2, default 1.0)"
                    },
                    "max_output_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens in response"
                    },
                    "previous_response_id": {
                        "type": "string",
                        "description": "ID of previous response for multi-turn conversations"
                    }
                },
                "required": ["model", "input"]
            }
        ),
        Tool(
            name="messages",
            description="Use Anthropic Messages API format. Native format for Claude models with support for system prompts, multi-turn conversations, and tool use.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model ID (e.g., 'claude-sonnet-4-20250514', 'gpt-4o')"
                    },
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"}
                            }
                        },
                        "description": "Array of message objects with role and content"
                    },
                    "system": {
                        "type": "string",
                        "description": "System prompt"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens in response (required for Claude)"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature (0-1, default 1.0)"
                    }
                },
                "required": ["model", "messages", "max_tokens"]
            }
        ),
        Tool(
            name="generate_content",
            description="Use Google generateContent API format. Native format for Gemini models with support for multimodal inputs and safety settings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model ID (e.g., 'gemini/gemini-1.5-pro', 'gpt-4o')"
                    },
                    "contents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "model"]},
                                "parts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "description": "Array of content objects with role and parts"
                    },
                    "system_instruction": {
                        "type": "string",
                        "description": "System instruction for the model"
                    },
                    "generation_config": {
                        "type": "object",
                        "description": "Generation config (temperature, maxOutputTokens, etc.)"
                    }
                },
                "required": ["model", "contents"]
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
    elif name == "responses":
        return await handle_responses(arguments)
    elif name == "messages":
        return await handle_messages(arguments)
    elif name == "generate_content":
        return await handle_generate_content(arguments)
    elif name == "compare":
        return await handle_compare(arguments)
    elif name == "models":
        return await handle_models(arguments)
    elif name == "recommend":
        return await handle_recommend(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_call(args: dict) -> list[TextContent]:
    """Call a single LLM model using chat completions format."""
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


async def handle_responses(args: dict) -> list[TextContent]:
    """Use OpenAI Responses API format."""
    try:
        import litellm
        
        model = args["model"]
        input_text = args["input"]
        instructions = args.get("instructions")
        temperature = args.get("temperature", 1.0)
        max_output_tokens = args.get("max_output_tokens")
        previous_response_id = args.get("previous_response_id")
        
        kwargs = {
            "model": model,
            "input": input_text,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens:
            kwargs["max_output_tokens"] = max_output_tokens
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
            
        response = await litellm.aresponses(**kwargs)
        
        # Extract text from response
        output_text = ""
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content') and item.content:
                    for content in item.content:
                        if hasattr(content, 'text'):
                            output_text += content.text
        
        result = {
            "id": getattr(response, 'id', None),
            "output": output_text,
            "usage": {
                "input_tokens": getattr(response.usage, 'input_tokens', None) if hasattr(response, 'usage') else None,
                "output_tokens": getattr(response.usage, 'output_tokens', None) if hasattr(response, 'usage') else None,
            }
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except ImportError:
        return [TextContent(type="text", text="Error: litellm not installed. Run: pip install litellm")]
    except AttributeError as e:
        # Fallback if responses API not available
        return [TextContent(type="text", text=f"Responses API may not be available for this model. Try using 'call' instead. Error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error with responses API: {str(e)}")]


async def handle_messages(args: dict) -> list[TextContent]:
    """Use Anthropic Messages API format."""
    try:
        import litellm
        
        model = args["model"]
        messages = args["messages"]
        system = args.get("system")
        max_tokens = args["max_tokens"]
        temperature = args.get("temperature", 1.0)
        
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
            
        # Use completion but format response like Messages API
        response = await litellm.acompletion(**kwargs)
        
        result = {
            "id": getattr(response, 'id', None),
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response.choices[0].message.content
                }
            ],
            "model": model,
            "stop_reason": response.choices[0].finish_reason,
            "usage": {
                "input_tokens": response.usage.prompt_tokens if response.usage else None,
                "output_tokens": response.usage.completion_tokens if response.usage else None,
            }
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except ImportError:
        return [TextContent(type="text", text="Error: litellm not installed. Run: pip install litellm")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error with messages API: {str(e)}")]


async def handle_generate_content(args: dict) -> list[TextContent]:
    """Use Google generateContent API format."""
    try:
        import litellm
        
        model = args["model"]
        contents = args["contents"]
        system_instruction = args.get("system_instruction")
        generation_config = args.get("generation_config", {})
        
        # Convert Gemini format to OpenAI format for LiteLLM
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        for content in contents:
            role = "assistant" if content.get("role") == "model" else "user"
            text_parts = []
            for part in content.get("parts", []):
                if "text" in part:
                    text_parts.append(part["text"])
            if text_parts:
                messages.append({"role": role, "content": " ".join(text_parts)})
        
        kwargs = {
            "model": model,
            "messages": messages,
        }
        if "temperature" in generation_config:
            kwargs["temperature"] = generation_config["temperature"]
        if "maxOutputTokens" in generation_config:
            kwargs["max_tokens"] = generation_config["maxOutputTokens"]
        if "topP" in generation_config:
            kwargs["top_p"] = generation_config["topP"]
            
        response = await litellm.acompletion(**kwargs)
        
        # Format response like Gemini API
        result = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": response.choices[0].message.content}
                        ],
                        "role": "model"
                    },
                    "finishReason": response.choices[0].finish_reason.upper() if response.choices[0].finish_reason else "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": response.usage.prompt_tokens if response.usage else None,
                "candidatesTokenCount": response.usage.completion_tokens if response.usage else None,
                "totalTokenCount": response.usage.total_tokens if response.usage else None,
            }
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except ImportError:
        return [TextContent(type="text", text="Error: litellm not installed. Run: pip install litellm")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error with generateContent API: {str(e)}")]


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
