"""
ONE-TIME SETUP — run this once, then save the printed IDs to your .env file.

Usage:
    python agent_setup.py
"""

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from .env

# 1. Create a persistent environment (the sandbox where tools execute)
environment = client.beta.environments.create(
    name="my-assistant-env",
    config={
        "type": "cloud",
        "networking": {"type": "unrestricted"},  # full internet access
    },
)
print(f"Environment ID: {environment.id}")

# 2. Create the agent (model/system/tools live here, not on the session)
agent = client.beta.agents.create(
    name="My Assistant",
    model="claude-opus-4-6",
    system="You are a helpful assistant. Be concise and clear.",
    tools=[
        {
            "type": "agent_toolset_20260401",  # bash, read, write, web_search, etc.
            "default_config": {"enabled": True},
        }
    ],
)
print(f"Agent ID:      {agent.id}")
print(f"Agent version: {agent.version}")
print()
print("Add these to your .env file:")
print(f"  AGENT_ID={agent.id}")
print(f"  ENVIRONMENT_ID={environment.id}")
