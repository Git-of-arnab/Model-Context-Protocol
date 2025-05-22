import sys
import asyncio
import json
import webbrowser
from typing import Optional,Sequence
from typing_extensions import Annotated, TypedDict
from contextlib import AsyncExitStack

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,trim_messages,ToolMessage,BaseMessage
from langgraph.managed import IsLastStep

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import boto3
import os
from dotenv import load_dotenv
load_dotenv()


aws_session = boto3.Session(aws_access_key_id=os.getenv('CLAUDE_KEY_ID'),
    aws_secret_access_key=os.getenv('CLAUDE_ACCESS_KEY'),
    region_name='us-east-1')

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

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

        mcp_tool_list = await load_mcp_tools(self.session)
        return mcp_tool_list
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep #This is mandatory for langgraph agent
    remaining_steps: int #This is mandatory for langgraph agent to avoid infinite loop

async def main():
    if len(sys.argv) < 2:
        print(f"Usage: python application_client.py {sys.argv[1]}")
        sys.exit(1)
        
    client = MCPClient()
    try:
        model = ChatBedrockConverse(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            aws_access_key_id=os.getenv("CLAUDE_KEY_ID"),
            aws_secret_access_key=os.getenv("CLAUDE_ACCESS_KEY"),
            region_name="us-east-1",
            temperature=0.3
        )
        mcp_tool_list = await client.connect_to_server(sys.argv[1])
        print(f"\nMCP Tool List using Langchain:\n{mcp_tool_list}")

        workflow = StateGraph(state_schema=State)
        agent = create_react_agent(model=model,tools=mcp_tool_list,state_schema=State,prompt='''Utilize the provided tools when required. The nearest place search tool and navigation tool already has the current location. 
                                   If you are using the navigation tool, then your response should strictly be {'url': the generated url,'text':'I am now opening the navigation tool for you'},without adding anything extra from your end.''')
        
        messages = []
        trimmer = trim_messages(
            max_tokens=8, #This is the count, last 5 messages will be kept as history
            strategy="last",
            token_counter=len, #"len" means it will keep messages on the count and not on tokens
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

        async def call_model(state: State):
            trimmed_messages = trimmer.invoke(state["messages"])
            print(f"Number of msgs in trimmed messages:{len(trimmed_messages)}")
            agent_response = await agent.ainvoke({"messages":trimmed_messages})

            return agent_response
        
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)
        memory = MemorySaver()
        graph_app = workflow.compile(checkpointer=memory)
        conversation = [
        {
            "type":"text",
            "text":"Navigate me to Stamford bridge"
        }]
        messages = messages + [HumanMessage(content=conversation)]
        final_response = await graph_app.ainvoke({"messages":messages},{"configurable": {"thread_id": "Test101"}})
        final_response = final_response['messages'][-1].content
        #print(agent_response['messages'][-1].content)
        if final_response.startswith('{'):
            dict_ = json.loads(final_response)
            print(f"Response converted to JSON:\n{dict_}\n{type(dict_)}")
            webbrowser.open_new_tab(dict_['url'])
        else:
            print(final_response)

    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())