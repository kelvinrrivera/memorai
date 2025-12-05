# mcp_server.py
import os
import asyncio
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mem0 import Memory
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse

# Configurar Mem0 Cliente
config = {
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "dbname": os.getenv("POSTGRES_DB", "mem0_db"),
        }
    }
}
m = Memory.from_config(config)

app = Server("mem0-mcp")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="add_memory",
            description="Add a memory or information to the database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Information to store"},
                    "user_id": {"type": "string", "description": "User ID"}
                },
                "required": ["data", "user_id"]
            }
        ),
        Tool(
            name="search_memory",
            description="Search for memories relevant to a query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "user_id": {"type": "string", "description": "User ID"}
                },
                "required": ["query", "user_id"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name, arguments):
    if name == "add_memory":
        result = m.add(arguments["data"], user_id=arguments["user_id"])
        return [TextContent(type="text", text=str(result))]
    
    if name == "search_memory":
        result = m.search(arguments["query"], user_id=arguments["user_id"])
        return [TextContent(type="text", text=str(result))]
    
    raise ValueError(f"Unknown tool: {name}")

# Configurar Servidor SSE (Starlette)
sse = SseServerTransport("/messages")

async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())

async def handle_messages(request):
    await sse.handle_post_message(request.scope, request.receive, request._send)

starlette_app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages", endpoint=handle_messages, methods=["POST"]),
    ]
)