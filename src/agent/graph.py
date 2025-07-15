
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, TypedDict
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from agent.tools.market_research import market_research_agent # Import the market research agent for Google Trends API 


class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str = "example"


async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime configuration to alter behavior.
    """
    configuration = config["configurable"]
    return {
        "data_D": json.dumps(structured_result.dict(), ensure_ascii=False),
        "messages": text_result
    }



graph = StateGraph(StateNDA)

# Register all agent nodes
graph.add_node("Agent_A", Agent_A)
graph.add_node("Agent_B", Agent_B)
graph.add_node("Agent_C", Agent_C)
graph.add_node("Agent_D", Agent_D)


graph.set_entry_point("Agent_A")

# After each child agent finishes, it returns to Agent_A
graph.add_edge("Agent_B", "Agent_A")
graph.add_edge("Agent_C", "Agent_A")
graph.add_edge("Agent_D", "Agent_A")


# Define supervisor logic: where to go next
graph.add_conditional_edges(
    "Agent_A",
    lambda state: state["next_node"],
    {
        "Agent_B": "Agent_B",
        "Agent_C": "Agent_C",
        "Agent_D": "Agent_D",
        "END": END
    }
)
