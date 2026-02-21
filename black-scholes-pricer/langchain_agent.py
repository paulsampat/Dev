#!/usr/bin/env python
"""
LangChain ReAct agent for Black-Scholes option pricing.
Connects to the local MCP server via stdio transport.

Usage:
    python langchain_agent.py                           # interactive loop
    python langchain_agent.py "Price a call: S=100..."  # single query

Requires ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

_HERE = Path(__file__).parent.resolve()

_MCP_CONFIG = {
    "black-scholes-pricer": {
        "command": sys.executable,
        "args": [str(_HERE / "mcp_server.py")],
        "transport": "stdio",
        "env": {"PYTHONPATH": str(_HERE)},
    }
}


async def run_agent(query: str) -> str:
    """Run a single query through the agent and return the final answer."""
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

    client = MultiServerMCPClient(_MCP_CONFIG)
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

    return result["messages"][-1].content


async def interactive_loop() -> None:
    """Start an interactive REPL-style loop."""
    print("Black-Scholes Agent  (type 'quit' to exit)\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        answer = await run_agent(query)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    if len(sys.argv) > 1:
        answer = asyncio.run(run_agent(" ".join(sys.argv[1:])))
        print(answer)
    else:
        asyncio.run(interactive_loop())
