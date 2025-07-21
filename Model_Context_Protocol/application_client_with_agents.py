import asyncio
from typing import Optional,Sequence
from contextlib import AsyncExitStack
from typing_extensions import Annotated, TypedDict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain.tools import tool
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,trim_messages,ToolMessage,BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.managed import IsLastStep
from langchain_aws import ChatBedrockConverse

import boto3
import os
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

model = ChatBedrockConverse(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    aws_access_key_id= os.getnev('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key= os.getenv('AWS_SECRET_KEY'),
    region_name="us-east-1",
    temperature=0.3
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: int

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic_bedrock = model

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        mcp_tool_list = await load_mcp_tools(self.session)
        return mcp_tool_list

    async def process_query(self, query: str, agent) -> str:
        """Process a query using Claude and available tools"""
        messages = {'messages':query}
        agent_response = await agent.ainvoke(messages)

        return agent_response['messages'][-1].content

    async def chat_loop(self,agent):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query,agent)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print(f"Usage: python application_client.py {sys.argv[1]}")
        sys.exit(1)
        
    client = MCPClient()
    try:
        mcp_tool_list = await client.connect_to_server(sys.argv[1])
        agent = create_react_agent(model=model,tools=mcp_tool_list,prompt="You are an weather expert with multiple tools at your disposal. Answer in a polite manner")
        await client.chat_loop(agent=agent)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
