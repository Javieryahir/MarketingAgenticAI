"""New LangGraph Agent.

This module defines a custom graph.
"""

try:
    from agent.graph import graph
except ImportError:
    from .graph import graph

__all__ = ["graph"]
