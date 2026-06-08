from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# tools
from langgraph.prebuilt import ToolNode, tools_condition

# MCP
from langchain_mcp_adapters.client import MultiServerMCPClient

import asyncio

load_dotenv()

# ---------------- STATE ----------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---------------- LLM ----------------
llm = ChatGroq(model="llama-3.1-8b-instant", verbose=True)

# ---------------- MCP CLIENT ----------------
client = MultiServerMCPClient(
    {
        "expense_tracker": {
            "transport": "streamable-http",
            "url": "http://localhost:8000/mcp"   # ✅ FIXED (IMPORTANT)
        },
        'Arith_server':{
            'transport': 'stdio',
            'command': 'uv',
            'args': ['run', 'C:\\My_Drive\\programming\\learning-MCP\\localToRemote\\arth.py']
        }
    }
)

# ---------------- GRAPH ----------------
graph = StateGraph(ChatState)

# ---------------- BUILD GRAPH ----------------
async def build_graph():
    # ✅ Load MCP tools
    tools = await client.get_tools()

    # ✅ Bind tools to LLM (you missed this earlier)
    llm_with_tools = llm.bind_tools(tools)

    # ---------------- CHAT NODE ----------------
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # ---------------- TOOL NODE ----------------
    tool_node = ToolNode(tools)

    # ---------------- GRAPH FLOW ----------------
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    app = graph.compile()
    return app

# ---------------- MAIN ----------------
async def main():
    chatbot = await build_graph()

    # ✅ Test input (better than "2*30")
    res = await chatbot.ainvoke({
        "messages": [
            HumanMessage(content="What is 2+2? using mcp tool")
        ]
    })

    print(res["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())