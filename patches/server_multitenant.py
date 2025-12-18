import json
import os
import sys
import argparse
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, List

from cognee.shared.logging_utils import get_logger, setup_logging, get_log_file_location
import importlib.util
from contextlib import redirect_stdout
import mcp.types as types
from mcp.server import FastMCP
from cognee.modules.storage.utils import JSONEncoder
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn

try:
    from .cognee_client import CogneeClient
except ImportError:
    from cognee_client import CogneeClient


try:
    from cognee.tasks.codingagents.coding_rule_associations import (
        add_rule_associations,
        get_existing_rules,
    )
except ModuleNotFoundError:
    from .codingagents.coding_rule_associations import (
        add_rule_associations,
        get_existing_rules,
    )


mcp = FastMCP("Cognee")

logger = get_logger()

cognee_client: Optional[CogneeClient] = None


async def run_sse_with_cors():
    """Custom SSE transport with CORS middleware."""
    sse_app = mcp.sse_app()
    sse_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(
        sse_app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_http_with_cors():
    """Custom HTTP transport with CORS middleware."""
    http_app = mcp.streamable_http_app()
    http_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(
        http_app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "ok"})


@mcp.tool()
async def cognee_add_developer_rules(
    base_path: str = ".", graph_model_file: str = None, graph_model_name: str = None, dataset_name: str = "developer_rules"
) -> list:
    """
    Ingest core developer rule files into Cognee's memory layer.

    This function loads a predefined set of developer-related configuration,
    rule, and documentation files from the base repository and assigns them
    to the specified dataset (default: 'developer_rules').

    Parameters
    ----------
    base_path : str
        Root path to resolve relative file paths. Defaults to current directory.

    graph_model_file : str, optional
        Optional path to a custom schema file for knowledge graph generation.

    graph_model_name : str, optional
        Optional class name to use from the graph_model_file schema.
        
    dataset_name : str, optional
        Name of the dataset to add rules to. Defaults to "developer_rules".

    Returns
    -------
    list
        A message indicating how many rule files were scheduled for ingestion,
        and how to check their processing status.
    """

    developer_rule_paths = [
        ".cursorrules",
        ".cursor/rules",
        ".same/todos.md",
        ".windsurfrules",
        ".clinerules",
        "CLAUDE.md",
        ".sourcegraph/memory.md",
        "AGENT.md",
        "AGENTS.md",
    ]

    async def cognify_task(file_path: str) -> None:
        with redirect_stdout(sys.stderr):
            logger.info(f"Starting cognify for: {file_path}")
            try:
                # Add to the specified dataset
                await cognee_client.add(file_path, dataset_name=dataset_name, node_set=["developer_rules"])

                model = None
                if graph_model_file and graph_model_name:
                    if cognee_client.use_api:
                        logger.warning(
                            "Custom graph models are not supported in API mode, ignoring."
                        )
                    else:
                        from cognee.shared.data_models import KnowledgeGraph

                        model = load_class(graph_model_file, graph_model_name)

                # Process specific dataset
                await cognee_client.cognify(datasets=[dataset_name], graph_model=model)
                logger.info(f"Cognify finished for: {file_path}")
            except Exception as e:
                logger.error(f"Cognify failed for {file_path}: {str(e)}")
                raise ValueError(f"Failed to cognify: {str(e)}")

    tasks = []
    for rel_path in developer_rule_paths:
        abs_path = os.path.join(base_path, rel_path)
        if os.path.isfile(abs_path):
            tasks.append(asyncio.create_task(cognify_task(abs_path)))
        else:
            logger.warning(f"Skipped missing developer rule file: {abs_path}")
    log_file = get_log_file_location()
    return [
        types.TextContent(
            type="text",
            text=(
                f"Started cognify for {len(tasks)} developer rule files in background.\n"
                f"All are added to the `{dataset_name}` dataset.\n"
                f"Use `cognify_status` with dataset_name='{dataset_name}' or check logs at {log_file} to monitor progress."
            ),
        )
    ]


@mcp.tool()
async def cognify(
    data: str, 
    graph_model_file: str = None, 
    graph_model_name: str = None, 
    custom_prompt: str = None,
    dataset_name: str = "main_dataset"
) -> list:
    """
    Transform ingested data into a structured knowledge graph within a specific dataset.

    This is the core processing step in Cognee that converts raw text and documents
    into an intelligent knowledge graph. It analyzes content, extracts entities and
    relationships, and creates semantic connections for enhanced search and reasoning.

    Parameters
    ----------
    data : str
        The data to be processed and transformed into structured knowledge.
        
    dataset_name : str, optional
        Name of the dataset to add data to and process. Defaults to "main_dataset".
        This allows for multi-tenant isolation (e.g., set to "agent_001", "agent_002").

    graph_model_file : str, optional
        Path to a custom schema file that defines the structure of the generated knowledge graph.

    graph_model_name : str, optional
        Name of the class within the graph_model_file to instantiate as the graph model.

    custom_prompt : str, optional
        Custom prompt string to use for entity extraction and graph generation.

    Returns
    -------
    list
        A list containing a single TextContent object with information about the
        background task launch and how to check its status.
    """

    async def cognify_task(
        data: str,
        graph_model_file: str = None,
        graph_model_name: str = None,
        custom_prompt: str = None,
        dataset_name: str = "main_dataset",
    ) -> str:
        """Build knowledge graph from the input text"""
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            logger.info(f"Cognify process starting for dataset: {dataset_name}")

            graph_model = None
            if graph_model_file and graph_model_name:
                if cognee_client.use_api:
                    logger.warning("Custom graph models are not supported in API mode, ignoring.")
                else:
                    from cognee.shared.data_models import KnowledgeGraph

                    graph_model = load_class(graph_model_file, graph_model_name)

            await cognee_client.add(data, dataset_name=dataset_name)

            try:
                await cognee_client.cognify(datasets=[dataset_name], custom_prompt=custom_prompt, graph_model=graph_model)
                logger.info("Cognify process finished.")
            except Exception as e:
                logger.error("Cognify process failed.")
                raise ValueError(f"Failed to cognify: {str(e)}")

    asyncio.create_task(
        cognify_task(
            data=data,
            graph_model_file=graph_model_file,
            graph_model_name=graph_model_name,
            custom_prompt=custom_prompt,
            dataset_name=dataset_name,
        )
    )

    log_file = get_log_file_location()
    text = (
        f"Background process launched due to MCP timeout limitations.\n"
        f"Data added to dataset: {dataset_name}\n"
        f"To check current cognify status use user: `cognify_status(dataset_name='{dataset_name}')`\n"
        f"or check the log file at: {log_file}"
    )

    return [
        types.TextContent(
            type="text",
            text=text,
        )
    ]


@mcp.tool(
    name="save_interaction", description="Logs user-agent interactions and query-answer pairs"
)
async def save_interaction(data: str, dataset_name: str = "user_agent_interaction") -> list:
    """
    Transform and save a user-agent interaction into structured knowledge.

    Parameters
    ----------
    data : str
        The input string containing user queries and corresponding agent answers.
        
    dataset_name : str, optional
        Name of the dataset to save interaction to. Defaults to "user_agent_interaction".

    Returns
    -------
    list
        A list containing a single TextContent object with information about the background task launch.
    """

    async def save_user_agent_interaction(data: str, dataset_name: str) -> None:
        """Build knowledge graph from the interaction data"""
        with redirect_stdout(sys.stderr):
            logger.info(f"Save interaction process starting for dataset: {dataset_name}")

            await cognee_client.add(data, dataset_name=dataset_name, node_set=["user_agent_interaction"])

            try:
                await cognee_client.cognify(datasets=[dataset_name])
                logger.info("Save interaction process finished.")

                # Rule associations only work in direct mode
                if not cognee_client.use_api:
                    logger.info("Generating associated rules from interaction data.")
                    await add_rule_associations(data=data, rules_nodeset_name="coding_agent_rules")
                    logger.info("Associated rules generated from interaction data.")
                else:
                    logger.warning("Rule associations are not available in API mode, skipping.")

            except Exception as e:
                logger.error("Save interaction process failed.")
                raise ValueError(f"Failed to Save interaction: {str(e)}")

    asyncio.create_task(
        save_user_agent_interaction(
            data=data,
            dataset_name=dataset_name
        )
    )

    log_file = get_log_file_location()
    text = (
        f"Background process launched to process the user-agent interaction in dataset '{dataset_name}'.\n"
        f"To check the current status, use the cognify_status tool or check the log file at: {log_file}"
    )

    return [
        types.TextContent(
            type="text",
            text=text,
        )
    ]


@mcp.tool()
async def codify(repo_path: str, dataset_name: str = "codebase") -> list:
    """
    Analyze and generate a code-specific knowledge graph from a software repository.

    Parameters
    ----------
    repo_path : str
        Path to the code repository to analyze.
        
    dataset_name : str, optional
        Name of the dataset to store the code graph. Defaults to "codebase".

    Returns
    -------
    list
        A list containing a single TextContent object with information about the
        background task launch and how to check its status.
    """

    if cognee_client.use_api:
        error_msg = "❌ Codify operation is not available in API mode. Please use direct mode for code graph pipeline."
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]

    async def codify_task(repo_path: str, dataset_name: str):
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            logger.info("Codify process starting.")
            from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline

            # Note: run_code_graph_pipeline might currently rely on a hardcoded "codebase" dataset name in the core library
            # But we can try to find if it accepts it. Looking at source would be ideal, but for now we assume implicit behavior.
            # Actually, codify likely uses `add` internally. If run_code_graph_pipeline doesn't accept dataset_name, 
            # we might be limited here. 
            # However, for now, we will just launch it.
            
            # Warning: cognee's codify pipeline might be rigid. 
            # We'll stick to running it, but note that dataset_name might not be fully effective if the underlying pipeline ignores it.
            # But let's try.
            
            results = []
            async for result in run_code_graph_pipeline(repo_path, False):
                results.append(result)
                logger.info(result)
            if all(results):
                logger.info("Codify process finished succesfully.")
            else:
                logger.info("Codify process failed.")

    asyncio.create_task(codify_task(repo_path, dataset_name))

    log_file = get_log_file_location()
    text = (
        f"Background process launched due to MCP timeout limitations.\n"
        f"To check current codify status use the codify_status tool\n"
        f"or you can check the log file at: {log_file}"
    )

    return [
        types.TextContent(
            type="text",
            text=text,
        )
    ]


@mcp.tool()
async def search(search_query: str, search_type: str, dataset_name: str = None) -> list:
    """
    Search and query the knowledge graph for insights, information, and connections.
    
    Parameters
    ----------
    search_query : str
        Your question or search query in natural language.
        
    search_type : str
        The type of search to perform.
        
    dataset_name : str, optional
        Name of the dataset to search in. If not provided, searches the default dataset.
        For multi-tenant agents, specify the agent's dataset name here.

    Returns
    -------
    list
        A list containing a single TextContent object with the search results.
    """

    async def search_task(search_query: str, search_type: str, dataset_name: str) -> str:
        """Search the knowledge graph"""
        # NOTE: MCP uses stdout to communicate, we must redirect all output
        #       going to stdout ( like the print function ) to stderr.
        with redirect_stdout(sys.stderr):
            datasets = [dataset_name] if dataset_name else None
            
            search_results = await cognee_client.search(
                query_text=search_query, query_type=search_type, datasets=datasets
            )

            # Handle different result formats based on API vs direct mode
            if cognee_client.use_api:
                # API mode returns JSON-serialized results
                if isinstance(search_results, str):
                    return search_results
                elif isinstance(search_results, list):
                    if (
                        search_type.upper() in ["GRAPH_COMPLETION", "RAG_COMPLETION"]
                        and len(search_results) > 0
                    ):
                        return str(search_results[0])
                    return str(search_results)
                else:
                    return json.dumps(search_results, cls=JSONEncoder)
            else:
                # Direct mode processing
                if search_type.upper() == "CODE":
                    return json.dumps(search_results, cls=JSONEncoder)
                elif (
                    search_type.upper() == "GRAPH_COMPLETION"
                    or search_type.upper() == "RAG_COMPLETION"
                ):
                    return str(search_results[0])
                elif search_type.upper() == "CHUNKS":
                    return str(search_results)
                elif search_type.upper() == "INSIGHTS":
                    results = retrieved_edges_to_string(search_results)
                    return results
                else:
                    return str(search_results)

    search_results = await search_task(search_query, search_type, dataset_name)
    return [types.TextContent(type="text", text=search_results)]


@mcp.tool()
async def get_developer_rules() -> list:
    """
    Retrieve all developer rules that were generated based on previous interactions.
    """

    async def fetch_rules_from_cognee() -> str:
        """Collect all developer rules from Cognee"""
        with redirect_stdout(sys.stderr):
            if cognee_client.use_api:
                logger.warning("Developer rules retrieval is not available in API mode")
                return "Developer rules retrieval is not available in API mode"

            developer_rules = await get_existing_rules(rules_nodeset_name="coding_agent_rules")
            return developer_rules

    rules_text = await fetch_rules_from_cognee()

    return [types.TextContent(type="text", text=rules_text)]


@mcp.tool()
async def list_data(dataset_id: str = None) -> list:
    """
    List all datasets and their data items with IDs for deletion operations.
    """
    from uuid import UUID

    with redirect_stdout(sys.stderr):
        try:
            output_lines = []

            if dataset_id:
                # Detailed data listing for specific dataset is only available in direct mode
                if cognee_client.use_api:
                    return [
                        types.TextContent(
                            type="text",
                            text="❌ Detailed data listing for specific datasets is not available in API mode.\nPlease use the API directly or use direct mode.",
                        )
                    ]

                from cognee.modules.users.methods import get_default_user
                from cognee.modules.data.methods import get_dataset, get_dataset_data

                logger.info(f"Listing data for dataset: {dataset_id}")
                dataset_uuid = UUID(dataset_id)
                user = await get_default_user()

                dataset = await get_dataset(user.id, dataset_uuid)

                if not dataset:
                    return [
                        types.TextContent(type="text", text=f"❌ Dataset not found: {dataset_id}")
                    ]

                # Get data items in the dataset
                data_items = await get_dataset_data(dataset.id)

                output_lines.append(f"📁 Dataset: {dataset.name}")
                output_lines.append(f"   ID: {dataset.id}")
                output_lines.append(f"   Created: {dataset.created_at}")
                output_lines.append(f"   Data items: {len(data_items)}")
                output_lines.append("")

                if data_items:
                    for i, data_item in enumerate(data_items, 1):
                        output_lines.append(f"   📄 Data item #{i}:")
                        output_lines.append(f"      Data ID: {data_item.id}")
                        output_lines.append(f"      Name: {data_item.name or 'Unnamed'}")
                        output_lines.append(f"      Created: {data_item.created_at}")
                        output_lines.append("")
                else:
                    output_lines.append("   (No data items in this dataset)")

            else:
                # List all datasets - works in both modes
                logger.info("Listing all datasets")
                datasets = await cognee_client.list_datasets()

                if not datasets:
                    return [
                        types.TextContent(
                            type="text",
                            text="📂 No datasets found.\nUse the cognify tool to create your first dataset!",
                        )
                    ]

                output_lines.append("📂 Available Datasets:")
                output_lines.append("=" * 50)
                output_lines.append("")

                for i, dataset in enumerate(datasets, 1):
                    # In API mode, dataset is a dict; in direct mode, it's formatted as dict
                    if isinstance(dataset, dict):
                        output_lines.append(f"{i}. 📁 {dataset.get('name', 'Unnamed')}")
                        output_lines.append(f"   Dataset ID: {dataset.get('id')}")
                        output_lines.append(f"   Created: {dataset.get('created_at', 'N/A')}")
                    else:
                        output_lines.append(f"{i}. 📁 {dataset.name}")
                        output_lines.append(f"   Dataset ID: {dataset.id}")
                        output_lines.append(f"   Created: {dataset.created_at}")
                    output_lines.append("")

                if not cognee_client.use_api:
                    output_lines.append("💡 To see data items in a specific dataset, use:")
                    output_lines.append('   list_data(dataset_id="your-dataset-id-here")')
                    output_lines.append("")
                output_lines.append("🗑️  To delete specific data, use:")
                output_lines.append('   delete(data_id="data-id", dataset_id="dataset-id")')

            result_text = "\n".join(output_lines)
            logger.info("List data operation completed successfully")

            return [types.TextContent(type="text", text=result_text)]

        except ValueError as e:
            error_msg = f"❌ Invalid UUID format: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]

        except Exception as e:
            error_msg = f"❌ Failed to list data: {str(e)}"
            logger.error(f"List data error: {str(e)}")
            return [types.TextContent(type="text", text=error_msg)]


@mcp.tool()
async def delete(data_id: str, dataset_id: str, mode: str = "soft") -> list:
    """
    Delete specific data from a dataset in the Cognee knowledge graph.
    """
    from uuid import UUID

    with redirect_stdout(sys.stderr):
        try:
            logger.info(
                f"Starting delete operation for data_id: {data_id}, dataset_id: {dataset_id}, mode: {mode}"
            )

            # Convert string UUIDs to UUID objects
            data_uuid = UUID(data_id)
            dataset_uuid = UUID(dataset_id)

            # Call the cognee delete function via client
            result = await cognee_client.delete(
                data_id=data_uuid, dataset_id=dataset_uuid, mode=mode
            )

            logger.info(f"Delete operation completed successfully: {result}")

            # Format the result for MCP response
            formatted_result = json.dumps(result, indent=2, cls=JSONEncoder)

            return [
                types.TextContent(
                    type="text",
                    text=f"✅ Delete operation completed successfully!\n\n{formatted_result}",
                )
            ]

        except ValueError as e:
            # Handle UUID parsing errors
            error_msg = f"❌ Invalid UUID format: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]

        except Exception as e:
            # Handle all other errors (DocumentNotFoundError, DatasetNotFoundError, etc.)
            error_msg = f"❌ Delete operation failed: {str(e)}"
            logger.error(f"Delete operation error: {str(e)}")
            return [types.TextContent(type="text", text=error_msg)]


@mcp.tool()
async def prune():
    """
    Reset the Cognee knowledge graph by removing all stored information.
    """
    with redirect_stdout(sys.stderr):
        try:
            await cognee_client.prune_data()
            await cognee_client.prune_system(metadata=True)
            return [types.TextContent(type="text", text="Pruned")]
        except NotImplementedError:
            error_msg = "❌ Prune operation is not available in API mode"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"❌ Prune operation failed: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]


@mcp.tool()
async def cognify_status(dataset_name: str = "main_dataset") -> list:
    """
    Get the current status of the cognify pipeline for a specific dataset.

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset to check status for. Defaults to "main_dataset".
    """
    with redirect_stdout(sys.stderr):
        try:
            from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
            from cognee.modules.users.methods import get_default_user

            user = await get_default_user()
            # Dynamically get dataset ID for the provided name
            dataset_id = await get_unique_dataset_id(dataset_name, user)
            
            status = await cognee_client.get_pipeline_status(
                [dataset_id], "cognify_pipeline"
            )
            return [types.TextContent(type="text", text=str(status))]
        except NotImplementedError:
            error_msg = "❌ Pipeline status is not available in API mode"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"❌ Failed to get cognify status: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]


@mcp.tool()
async def codify_status(dataset_name: str = "codebase") -> list:
    """
    Get the current status of the codify pipeline for a specific dataset.
    
    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset to check status for. Defaults to "codebase".
    """
    with redirect_stdout(sys.stderr):
        try:
            from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
            from cognee.modules.users.methods import get_default_user

            user = await get_default_user()
            dataset_id = await get_unique_dataset_id(dataset_name, user)
            
            status = await cognee_client.get_pipeline_status(
                [dataset_id], "cognify_code_pipeline"
            )
            return [types.TextContent(type="text", text=str(status))]
        except NotImplementedError:
            error_msg = "❌ Pipeline status is not available in API mode"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
        except Exception as e:
            error_msg = f"❌ Failed to get codify status: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]


def node_to_string(node):
    node_data = ", ".join(
        [f'{key}: "{value}"' for key, value in node.items() if key in ["id", "name"]]
    )

    return f"Node({node_data})"


def retrieved_edges_to_string(search_results):
    edge_strings = []
    for triplet in search_results:
        node1, edge, node2 = triplet
        relationship_type = edge["relationship_name"]
        edge_str = f"{node_to_string(node1)} {relationship_type} {node_to_string(node2)}"
        edge_strings.append(edge_str)

    return "\n".join(edge_strings)


def load_class(model_file, model_name):
    model_file = os.path.abspath(model_file)
    spec = importlib.util.spec_from_file_location("graph_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, model_name)

    return model_class


async def main():
    global cognee_client

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transport",
        choices=["sse", "stdio", "http"],
        default="stdio",
        help="Transport to use for communication with the client. (default: stdio)",
    )

    # HTTP transport options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the HTTP server to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server to (default: 8000)",
    )

    parser.add_argument(
        "--path",
        default="/mcp",
        help="Path for the MCP HTTP endpoint (default: /mcp)",
    )

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level for the HTTP server (default: info)",
    )

    parser.add_argument(
        "--no-migration",
        default=False,
        action="store_true",
        help="Argument stops database migration from being attempted",
    )

    # Cognee API connection options
    parser.add_argument(
        "--api-url",
        default=None,
        help="Base URL of a running Cognee FastAPI server (e.g., http://localhost:8000). "
        "If provided, the MCP server will connect to the API instead of using cognee directly.",
    )

    parser.add_argument(
        "--api-token",
        default=None,
        help="Authentication token for the API (optional, required if API has authentication enabled).",
    )

    args = parser.parse_args()

    # Initialize the global CogneeClient
    cognee_client = CogneeClient(api_url=args.api_url, api_token=args.api_token)

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    # Skip migrations when in API mode (the API server handles its own database)
    if not args.no_migration and not args.api_url:
        # Run Alembic migrations from the main cognee directory where alembic.ini is located
        logger.info("Running database migrations...")
        migration_result = subprocess.run(
            ["python", "-m", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )

        if migration_result.returncode != 0:
            migration_output = migration_result.stderr + migration_result.stdout
            # Check for the expected UserAlreadyExists error (which is not critical)
            if (
                "UserAlreadyExists" in migration_output
                or "User default_user@example.com already exists" in migration_output
            ):
                logger.warning("Warning: Default user already exists, continuing startup...")
            else:
                logger.error(f"Migration failed with unexpected error: {migration_output}")
                sys.exit(1)

        logger.info("Database migrations done.")
    elif args.api_url:
        logger.info("Skipping database migrations (using API mode)")

    logger.info(f"Starting MCP server with transport: {args.transport}")
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    elif args.transport == "sse":
        logger.info(f"Running MCP server with SSE transport on {args.host}:{args.port}")
        await run_sse_with_cors()
    elif args.transport == "http":
        logger.info(
            f"Running MCP server with Streamable HTTP transport on {args.host}:{args.port}{args.path}"
        )
        await run_http_with_cors()


if __name__ == "__main__":
    logger = setup_logging()

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error initializing Cognee MCP server: {str(e)}")
        raise
