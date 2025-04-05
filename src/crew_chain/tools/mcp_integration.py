"""
Model Context Protocol (MCP) integration for Crew Chain.

This module provides integration with the Model Context Protocol (MCP),
allowing the trading system to leverage external AI services and tools.
"""
import json
import logging
import requests
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    name: str = Field(..., description="Name of the MCP server")
    url: str = Field(..., description="URL of the MCP server")
    api_key: Optional[str] = Field(None, description="API key for the MCP server")
    tools: List[str] = Field(default_factory=list, description="List of tools available on the server")

class MCPTool(BaseModel):
    """Representation of an MCP tool."""
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of the tool")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool")
    server: str = Field(..., description="Name of the server this tool belongs to")

class MCPClient:
    """Client for interacting with MCP servers."""
    
    def __init__(self, servers: List[MCPServerConfig]):
        """Initialize the MCP client with a list of server configurations."""
        self.servers = {server.name: server for server in servers}
        self.tools = self._discover_tools()
        
    def _discover_tools(self) -> Dict[str, MCPTool]:
        """Discover available tools from all configured servers."""
        tools = {}
        
        for server_name, server_config in self.servers.items():
            try:
                response = self._send_request(
                    server_config,
                    "GET",
                    "/tools"
                )
                
                if response.status_code == 200:
                    server_tools = response.json().get("tools", [])
                    for tool in server_tools:
                        tool_name = tool.get("name")
                        if tool_name:
                            tools[tool_name] = MCPTool(
                                name=tool_name,
                                description=tool.get("description", ""),
                                parameters=tool.get("parameters", {}),
                                server=server_name
                            )
                    logger.info(f"Discovered {len(server_tools)} tools from {server_name}")
                else:
                    logger.warning(f"Failed to discover tools from {server_name}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error discovering tools from {server_name}: {e}")
                
        return tools
    
    def get_available_tools(self) -> List[str]:
        """Get a list of available tool names."""
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[MCPTool]:
        """Get information about a specific tool."""
        return self.tools.get(tool_name)
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with the given parameters."""
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        server_config = self.servers.get(tool.server)
        if not server_config:
            raise ValueError(f"Server '{tool.server}' not found")
        
        request_data = {
            "name": tool_name,
            "parameters": parameters
        }
        
        try:
            response = self._send_request(
                server_config,
                "POST",
                "/execute",
                json=request_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error executing tool {tool_name}: {response.status_code} - {response.text}")
                return {"error": response.text}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}
    
    def _send_request(
        self, 
        server_config: MCPServerConfig, 
        method: str, 
        endpoint: str, 
        **kwargs
    ) -> requests.Response:
        """Send a request to an MCP server."""
        url = f"{server_config.url.rstrip('/')}{endpoint}"
        headers = kwargs.pop("headers", {})
        
        if server_config.api_key:
            headers["Authorization"] = f"Bearer {server_config.api_key}"
        
        headers["Content-Type"] = "application/json"
        kwargs["headers"] = headers
        
        return requests.request(method, url, **kwargs)

class MCPToolWrapper:
    """Wrapper for an MCP tool to use within CrewAI."""
    
    def __init__(self, client: MCPClient, tool_name: str):
        """Initialize the MCP tool wrapper."""
        self.client = client
        self.tool_name = tool_name
        self.tool_info = client.get_tool_info(tool_name)
        
        if not self.tool_info:
            raise ValueError(f"Tool '{tool_name}' not found")
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with the given parameters."""
        return self.client.execute_tool(self.tool_name, kwargs)
    
    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return self.tool_name
    
    @property
    def description(self) -> str:
        """Get the description of the tool."""
        return self.tool_info.description if self.tool_info else ""

def load_mcp_config(config_path: str) -> List[MCPServerConfig]:
    """Load MCP server configurations from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return [MCPServerConfig(**server_config) for server_config in config_data]
    except Exception as e:
        logger.error(f"Error loading MCP config from {config_path}: {e}")
        return [] 