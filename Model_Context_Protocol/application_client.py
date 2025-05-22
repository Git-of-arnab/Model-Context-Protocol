import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import boto3
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

session = boto3.Session(aws_access_key_id='AKIA5FTZBAU7TV43RUUP',
    aws_secret_access_key='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',
    region_name='us-east-1')

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic_bedrock = session.client("bedrock-runtime", region_name='us-east-1')

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
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        print(f"\nInput Schema:{[tool.inputSchema for tool in tools]}")

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": [{"text": query}]
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{"toolSpec":
            { 
            "name": tool.name,
            "description": tool.description,
            "inputSchema": {"json":tool.inputSchema}
            }
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic_bedrock.converse(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            messages=messages,
            inferenceConfig={"maxTokens": 512, "temperature": 0.3, "topP": 0.9},
            toolConfig = {"tools":available_tools}
        )
        content_dict = {k:v for elements in response['output']['message']['content'] for k,v in elements.items()}
        print(f"Formatted LLM Response:\n{content_dict}")

        # Process response and handle tool calls
        final_text = []
        if response['stopReason'] != 'tool_use':
            final_text.append(content_dict['text'])
        else:
            tool_name = content_dict['toolUse']['name']
            tool_args = content_dict['toolUse']['input']
                
            # Execute tool call
            result = await self.session.call_tool(tool_name, tool_args)
            # print(f"\nTool Call Result:\n{result}")
            # print(f"\nTool Call Result Text:\n{result.content[0].text}")
            final_text.append(f"\n[Calling tool {tool_name} with args {tool_args}]")

            # Continue conversation with tool results
            if content_dict['text']:
                messages.append({
                    "role": "assistant",
                    "content": [{"text":content_dict['text']}]
                })
            messages.append({
                "role": "user", 
                "content": [{"text":result.content[0].text}]
            })

            # Get next response from Claude
            response = self.anthropic_bedrock.converse(
                modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
                messages=messages,
                inferenceConfig={"maxTokens": 512, "temperature": 0.3, "topP": 0.9},
            )

            final_text.append(response['output']['message']['content'][0]['text'])

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
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
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())