"""Minimal stdio MCP server exposing basic math operations.

Run directly (stdio transport by default) or through the chatbot backend.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math-server")


@mcp.tool()
def add(first_num: float, second_num: float) -> float:
    """Add two numbers."""
    return first_num + second_num


@mcp.tool()
def sub(first_num: float, second_num: float) -> float:
    """Subtract the second number from the first."""
    return first_num - second_num


@mcp.tool()
def mul(first_num: float, second_num: float) -> float:
    """Multiply two numbers."""
    return first_num * second_num


@mcp.tool()
def div(first_num: float, second_num: float) -> float:
    """Divide the first number by the second."""
    if second_num == 0:
        raise ValueError("Division by zero is not allowed")
    return first_num / second_num


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the given exponent."""
    return base**exponent


if __name__ == "__main__":
    mcp.run(transport="stdio")
