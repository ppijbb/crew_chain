#!/usr/bin/env python
"""
Demo script to show how to use MCP tools in Crew Chain.
"""
import os
import sys
import logging
from pprint import pprint

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.crew_chain.tools.mcp_integration import MCPClient, MCPToolWrapper, load_mcp_config
from langchain.tools import Tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def main():
    """Run the MCP tools demo."""
    logger.info("Starting MCP tools demo")
    
    # Load MCP server configurations
    mcp_config_path = "src/crew_chain/config/mcp_servers.json"
    mcp_server_configs = load_mcp_config(mcp_config_path)
    
    if not mcp_server_configs:
        logger.error(f"No MCP server configurations found in {mcp_config_path}")
        return
    
    logger.info(f"Loaded {len(mcp_server_configs)} MCP server configurations")
    
    # Initialize MCP client
    mcp_client = MCPClient(mcp_server_configs)
    
    # Get available tools
    available_tools = mcp_client.get_available_tools()
    logger.info(f"Discovered {len(available_tools)} MCP tools:")
    for tool in available_tools:
        tool_info = mcp_client.get_tool_info(tool)
        logger.info(f"  - {tool}: {tool_info.description if tool_info else 'No description'}")
    
    # Demo: Use time MCP tool if available
    if "get_current_time" in available_tools:
        logger.info("\nDemonstrating 'get_current_time' tool:")
        time_tool = MCPToolWrapper(mcp_client, "get_current_time")
        try:
            result = time_tool()
            print(f"Current time: {result}")
        except Exception as e:
            logger.error(f"Error using time tool: {e}")
    
    # Demo: Use search tool if available
    if "search" in available_tools:
        logger.info("\nDemonstrating 'search' tool:")
        search_tool = MCPToolWrapper(mcp_client, "search")
        try:
            result = search_tool(query="cryptocurrency market trends 2024")
            print("Search results:")
            pprint(result)
        except Exception as e:
            logger.error(f"Error using search tool: {e}")
    
    # Demo: Create a LangChain tool from an MCP tool wrapper
    if "scrape_webpage" in available_tools:
        logger.info("\nDemonstrating LangChain Tool from MCP wrapper:")
        scrape_tool_wrapper = MCPToolWrapper(mcp_client, "scrape_webpage")
        scrape_tool = Tool(
            name="scrape_webpage",
            description=scrape_tool_wrapper.description,
            func=scrape_tool_wrapper
        )
        
        print(f"Tool name: {scrape_tool.name}")
        print(f"Tool description: {scrape_tool.description}")
        
        try:
            result = scrape_tool.run("https://coinmarketcap.com/")
            print("Webpage scraping results (truncated):")
            print(result[:500] + "..." if len(result) > 500 else result)
        except Exception as e:
            logger.error(f"Error using scrape_webpage tool: {e}")
    
    logger.info("\nMCP tools demo completed")

if __name__ == "__main__":
    main() 