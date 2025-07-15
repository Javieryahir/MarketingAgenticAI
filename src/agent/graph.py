
from __future__ import annotations
import json
import os
import uuid
import datetime
from typing import List, Dict, Any, Optional, Literal, Annotated
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")

# --- 1. Define the CNR Data Structures with Pydantic ---
# Enhanced with descriptions to guide the LLM in tool calling.

# Nested models must be defined before they are referenced.
class PersonaProfile(BaseModel):
    target_demographics: str = Field(..., description="The persona's age, gender, income, location, etc.")
    psychographics: str = Field(..., description="Interests, lifestyle, personality traits, values.")
    behavior_and_habits: str = Field(..., description="Online behavior, purchase patterns, content preferences.")
    challenges_or_pain_points: str = Field(..., description="What problems or frustrations they face.")
    effective_messaging_strategy: str = Field(..., description="The tone, format, or message that resonates with them.")
    summary_paragraph: str = Field(..., description="A short, fictional paragraph describing a person representing this segment.")

class PacketHeader(BaseModel):
    packetID: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this reasoning packet.")
    agentID: str = Field(..., description="The unique identifier for the agent generating this packet.")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(), description="The UTC timestamp of when the packet was created.")
    decisionEpoch: Literal["PLANNING", "EXECUTION", "ANALYSIS"] = Field(..., description="The operational phase during which the decision is being made.")

class CoreDecision(BaseModel):
    decisionStatement: str = Field(..., description="A clear, concise statement of the decision made.")
    decisionPayload: Dict[str, Any] = Field(..., description="A structured dictionary containing the actual data of the decision (e.g., the next agent to call, the market research results).")
    confidenceScore: float = Field(..., ge=0.0, le=1.0, description="The agent's confidence in its decision, from 0.0 to 1.0.")

class NarrativeLayer(BaseModel):
    rationalization: str = Field(..., description="A step-by-step explanation of the reasoning process that led to the decision.")
    pathwayContext: Dict[str, Any] = Field(..., description="A summary of the key inputs and data points from the state or log trail that were most influential in this decision.")
    governingIntuition: Optional[List[str]] = Field(None, description="Any underlying heuristics, principles, or 'gut feelings' that guided the reasoning.")

class CausalLayer(BaseModel):
    counterfactuals: List[Dict[str, Any]] = Field(..., description="A list of at least one counterfactual scenario. Each dictionary should contain an 'alternative' (what could have been different) and an 'outcome' (the expected result).")
    interventionsAnalysis: Optional[Dict[str, Any]] = Field(None, description="Analysis of potential interventions and their likely impact.")
    confounderAlerts: Optional[List[Dict[str, Any]]] = Field(None, description="Identification of potential confounding variables that could affect the decision.")

class RobustnessLayer(BaseModel):
    stabilityAnalysis: Optional[Dict[str, Any]] = Field(None, description="Analysis of how stable the decision is to small changes in input data.")
    uncertaintyDistribution: Optional[Dict[str, Any]] = Field(None, description="Where the key uncertainties in the reasoning process lie.")
    dataSufficiencyWarning: Optional[str] = Field(None, description="A warning if the available data was insufficient to make a high-confidence decision.")

class ReasoningPacket(BaseModel):
    """The main tool for agents to structure their reasoning and decisions."""
    packetHeader: PacketHeader
    coreDecision: CoreDecision
    narrativeLayer: NarrativeLayer
    causalLayer: CausalLayer
    robustnessLayer: RobustnessLayer
    learningLayer: Dict[str, Any] = Field(default_factory=dict, description="A field for capturing feedback or insights for future improvements.")

# NEW, simpler Pydantic model for the Supervisor's output
class SupervisorDecision(BaseModel):
    """A simpler tool for the supervisor to output its routing decision and reasoning."""
    next_node: Literal["Agent_B", "Agent_C", "Agent_D", "END"] = Field(..., description="The next agent node to call in the workflow, based on the log trail.")
    rationalization: str = Field(..., description="The detailed justification for why this is the correct next step.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this routing decision.")

# --- NEW, simpler Pydantic models for specialist agent outputs ---
class MarketResearchData(BaseModel):
    """A simpler data structure for the Market Research Agent's core output."""
    trends: List[str] = Field(..., description="A list of current and emerging market trends.")
    competitors: List[str] = Field(..., description="A list of key competitors and their strategies.")
    insights_summary: str = Field(..., description="A clear, concise summary of the overall findings.")
    recommended_positioning: Optional[str] = Field(None, description="A suggestion for how the product should be positioned.")

class AudienceAnalysisData(BaseModel):
    """A simpler data structure for the Audience Agent's core output."""
    personas: List[PersonaProfile] = Field(..., description="A list of 1-3 detailed audience persona profiles.")
    segmentation_strategy: str = Field(..., description="The strategy for how the audience can be segmented.")
    preferred_channels: List[str] = Field(..., description="A list of 2-5 channels where the campaign should focus.")

class ContentStrategyData(BaseModel):
    """A simpler data structure for the Content Strategy Agent's core output."""
    content_items: List[dict] = Field(..., description="A list of specific content items, each a dictionary with keys: description, content_type, campaign_theme, upload_date, upload_hour.")

# Rebuild the model to ensure forward references are resolved, as recommended by Pydantic.
AudienceAnalysisData.model_rebuild()

# --- 2. Update the main Graph State TypedDict ---

def add_to_log(left: list, right: list) -> list:
    """Reducer to append items to the log trail."""
    return left + right

class StateNDA(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log_trail: Annotated[list, add_to_log]
    data_A: Optional[str]
    data_B: Optional[str]
    data_C: Optional[str]
    data_D: Optional[str]

# --- 3. Implement Agent Nodes using the Assembler Pattern ---

llm = ChatOpenAI(model="gpt-4o-mini")

def supervisor_node(state: StateNDA, config: RunnableConfig):
    """
    An assembler-pattern node for the supervisor. It prompts the LLM for a simple
    decision and then assembles that decision into a full ReasoningPacket.
    This is more robust than asking the LLM to generate the complex nested packet.
    """
    supervisor_prompt_text = """
    You are the Supervisor Agent. Your task is to coordinate a team by analyzing the progress update to decide the next logical step.

    Workflow Stages:
    1. Start with `Agent_B` (Market Research) - if MarketResearchAgent hasn't run yet.
    2. Proceed to `Agent_C` (Audience Analysis) - if AudienceAgent hasn't run yet but MarketResearchAgent has.
    3. Then, use `Agent_D` (Content Strategy) - if ContentStrategyAgent hasn't run yet but both previous agents have.
    4. Once all specialist agents have run (MarketResearchAgent, AudienceAgent, ContentStrategyAgent), the next step is `END`.

    Look at the progress update in the message to see which agents have completed their work. Route to the next agent that hasn't run yet. Call the `SupervisorDecision` tool to state your decision.
    """
    llm_with_tool = llm.bind_tools([SupervisorDecision])
    prompt = ChatPromptTemplate.from_messages([("system", supervisor_prompt_text), ("placeholder", "{messages}")])
    agent_runnable = prompt | llm_with_tool
    
    # Clean state to avoid orphaned tool calls but include progress summary
    completed_agents = []
    log_trail = state.get("log_trail", [])
    for packet in log_trail:
        if hasattr(packet, 'packetHeader'):
            agent_id = packet.packetHeader.agentID
            if agent_id not in completed_agents:
                completed_agents.append(agent_id)
    
    # Create a summary message about completed work
    original_brief = state.get('messages', [{}])[0].content if state.get('messages') else 'No brief provided'
    if completed_agents:
        summary_content = f"Campaign brief: {original_brief}\n\nProgress update: The following agents have completed their work: {', '.join(completed_agents)}. Log trail contains {len(log_trail)} reasoning packets."
    else:
        summary_content = f"Campaign brief: {original_brief}\n\nProgress update: No agents have completed work yet. Starting workflow."
    
    clean_state = {
        "messages": [HumanMessage(content=summary_content)],
        "log_trail": log_trail
    }
    
    simple_decision = None
    for i in range(3):
        try:
            result: AIMessage = agent_runnable.invoke(clean_state)
            if not result.tool_calls:
                raise ValueError("Supervisor failed to call the SupervisorDecision tool.")
            tool_call_args = result.tool_calls[0]['args']
            simple_decision = SupervisorDecision.model_validate(tool_call_args)
            
            # Create the required ToolMessage to complete the handshake
            tool_response = ToolMessage(
                content=f"Successfully processed supervisor decision: routing to {simple_decision.next_node}",
                tool_call_id=result.tool_calls[0]['id'],
            )
            break 
        except Exception as e:
            if i < 2:
                # Don't add error messages to conversation state - just retry
                continue
            else:
                raise ValueError(f"Agent 'Supervisor' failed after multiple attempts. Last error: {e}")

    # Assemble the full ReasoningPacket using the validated simple decision
    reasoning_packet = ReasoningPacket(
        packetHeader=PacketHeader(agentID="Supervisor", decisionEpoch="PLANNING"),
        coreDecision=CoreDecision(
            decisionStatement=f"Routing to node '{simple_decision.next_node}'",
            decisionPayload={"next_node": simple_decision.next_node},
            confidenceScore=simple_decision.confidence,
        ),
        narrativeLayer=NarrativeLayer(
            rationalization=simple_decision.rationalization,
            pathwayContext={"inputs_reviewed": [msg.type for msg in state.get("messages", [])]},
        ),
        causalLayer=CausalLayer(
            counterfactuals=[{
                "alternative": f"A different routing decision was considered but rejected.",
                "outcome": "Proceeding out of order would lead to a suboptimal plan."
            }]
        ),
        robustnessLayer=RobustnessLayer(),
        learningLayer={}
    )

    return {
        "log_trail": [reasoning_packet],
        "messages": [result, tool_response], # Add both AI request and our response
        "data_A": json.dumps(reasoning_packet.model_dump())
    }

def market_research_node(state: StateNDA, config: RunnableConfig):
    """An assembler-pattern node for the Market Research Agent."""
    prompt_text = """
    You are the Market Research Agent. Analyze the initial request to provide a market overview.
    Call the `MarketResearchData` tool to structure your findings.
    """
    llm_with_tool = llm.bind_tools([MarketResearchData])
    prompt = ChatPromptTemplate.from_messages([("system", prompt_text), ("placeholder", "{messages}")])
    agent_runnable = prompt | llm_with_tool
    
    # Clean state to avoid orphaned tool calls
    clean_state = {
        "messages": [msg for msg in state.get("messages", []) if msg.type == "human"],
        "log_trail": state.get("log_trail", [])
    }

    core_data = None
    for i in range(3):
        try:
            result: AIMessage = agent_runnable.invoke(clean_state)
            if not result.tool_calls: raise ValueError("Market Research Agent failed to call tool.")
            tool_call_args = result.tool_calls[0]['args']
            core_data = MarketResearchData.model_validate(tool_call_args)
            break
        except Exception as e:
            if i < 2: 
                # Don't add error messages to conversation state - just retry
                continue
            else: raise ValueError(f"Agent 'Market Research' failed after multiple attempts. Last error: {e}")

    tool_response = ToolMessage(content="Successfully generated market research.", tool_call_id=result.tool_calls[0]['id'])
    reasoning_packet = ReasoningPacket(
        packetHeader=PacketHeader(agentID="MarketResearchAgent", decisionEpoch="EXECUTION"),
        coreDecision=CoreDecision(decisionStatement="Market analysis complete.", decisionPayload=core_data.model_dump(), confidenceScore=0.8),
        narrativeLayer=NarrativeLayer(rationalization="Analysis based on keywords in the initial brief.", pathwayContext={"inputs_reviewed": ["initial_brief"]}),
        causalLayer=CausalLayer(counterfactuals=[{"alternative": "If the brief mentioned a different region", "outcome": "The competitive landscape would change."}]),
        robustnessLayer=RobustnessLayer()
    )
    return {"log_trail": [reasoning_packet], "messages": [result, tool_response], "data_B": json.dumps(core_data.model_dump())}

def audience_agent_node(state: StateNDA, config: RunnableConfig):
    """An assembler-pattern node for the Audience Agent."""
    prompt_text = """
    You are the Audience Research Agent. Use the initial brief and market research data in the log trail to create detailed audience personas.
    Call the `AudienceAnalysisData` tool to structure your response.
    """
    llm_with_tool = llm.bind_tools([AudienceAnalysisData])
    prompt = ChatPromptTemplate.from_messages([("system", prompt_text), ("placeholder", "{messages}")])
    agent_runnable = prompt | llm_with_tool
    
    # Clean state to avoid orphaned tool calls
    clean_state = {
        "messages": [msg for msg in state.get("messages", []) if msg.type == "human"],
        "log_trail": state.get("log_trail", [])
    }

    core_data = None
    for i in range(3):
        try:
            result: AIMessage = agent_runnable.invoke(clean_state)
            if not result.tool_calls:
                raise ValueError("Audience Agent failed to call the AudienceAnalysisData tool.")
            tool_call_args = result.tool_calls[0]['args']
            core_data = AudienceAnalysisData.model_validate(tool_call_args)
            break
        except Exception as e:
            if i < 2:
                # Don't add error messages to conversation state - just retry
                continue
            else:
                raise ValueError(f"Agent 'Audience Agent' failed after multiple attempts. Last error: {e}")

    # Assemble the full ReasoningPacket
    tool_response = ToolMessage(content="Successfully generated audience analysis.", tool_call_id=result.tool_calls[0]['id'])
    reasoning_packet = ReasoningPacket(
        packetHeader=PacketHeader(agentID="AudienceAgent", decisionEpoch="EXECUTION"),
        coreDecision=CoreDecision(
            decisionStatement="Audience profiling complete.",
            decisionPayload=core_data.model_dump(),
            confidenceScore=0.85 
        ),
        narrativeLayer=NarrativeLayer(
            rationalization="Personas were derived from market trends and the initial product description.",
            pathwayContext={"inputs_reviewed": ["log_trail[Agent_B]", "initial_brief"]}
        ),
        causalLayer=CausalLayer(counterfactuals=[{"alternative": "If market was B2B", "outcome": "Personas would be job roles, not lifestyle archetypes."}]),
        robustnessLayer=RobustnessLayer()
    )

    return {
        "log_trail": [reasoning_packet],
        "messages": [result, tool_response],
        "data_C": json.dumps(core_data.model_dump())
    }

def content_strategy_node(state: StateNDA, config: RunnableConfig):
    """An assembler-pattern node for the Content Strategy Agent."""
    prompt_text = """
    You are the Content Strategy Agent. Use all information in the log trail to create a content plan.
    Call the `ContentStrategyData` tool with a 'content_items' field containing a list of content items.
    Each content item should be a dictionary with keys: description, content_type, campaign_theme, upload_date, upload_hour.
    """
    llm_with_tool = llm.bind_tools([ContentStrategyData])
    prompt = ChatPromptTemplate.from_messages([("system", prompt_text), ("placeholder", "{messages}")])
    agent_runnable = prompt | llm_with_tool
    
    # Clean state to avoid orphaned tool calls
    clean_state = {
        "messages": [msg for msg in state.get("messages", []) if msg.type == "human"],
        "log_trail": state.get("log_trail", [])
    }

    core_data = None
    for i in range(3):
        try:
            result: AIMessage = agent_runnable.invoke(clean_state)
            if not result.tool_calls: raise ValueError("Content Strategy Agent failed to call tool.")
            tool_call_args = result.tool_calls[0]['args']
            core_data = ContentStrategyData.model_validate(tool_call_args)
            break
        except Exception as e:
            if i < 2: 
                # Don't add error messages to conversation state - just retry
                continue
            else: raise ValueError(f"Agent 'Content Strategy' failed after multiple attempts. Last error: {e}")

    tool_response = ToolMessage(content="Successfully generated content strategy.", tool_call_id=result.tool_calls[0]['id'])
    reasoning_packet = ReasoningPacket(
        packetHeader=PacketHeader(agentID="ContentStrategyAgent", decisionEpoch="EXECUTION"),
        coreDecision=CoreDecision(decisionStatement="Content strategy generation complete.", decisionPayload=core_data.model_dump(), confidenceScore=0.9),
        narrativeLayer=NarrativeLayer(rationalization="Content items are based on audience personas and market trends.", pathwayContext={"inputs_reviewed": ["log_trail[Agent_B]", "log_trail[Agent_C]"]}),
        causalLayer=CausalLayer(counterfactuals=[{"alternative": "If the budget were smaller", "outcome": "The number of high-production video items would be reduced."}]),
        robustnessLayer=RobustnessLayer()
    )
    return {"log_trail": [reasoning_packet], "messages": [result, tool_response], "data_D": json.dumps(core_data.model_dump())}

# --- 4. Define the Graph and Router ---

graph = StateGraph(StateNDA)

graph.add_node("Agent_A", supervisor_node)
graph.add_node("Agent_B", market_research_node)
graph.add_node("Agent_C", audience_agent_node)
graph.add_node("Agent_D", content_strategy_node)

def router(state: StateNDA) -> Literal["Agent_B", "Agent_C", "Agent_D", "END"]:
    """Reads the supervisor's decision from data_A and routes accordingly."""
    data_A_str = state.get("data_A")
    if data_A_str:
        try:
            packet_dict = json.loads(data_A_str)
            next_node = packet_dict.get('coreDecision', {}).get('decisionPayload', {}).get('next_node')
            if next_node in ["Agent_B", "Agent_C", "Agent_D", "END"]:
                return next_node
        except (json.JSONDecodeError, AttributeError):
            return "END" # Default to end on parsing error
    return "END"

graph.add_conditional_edges(
    "Agent_A",
    router,
    {"Agent_B": "Agent_B", "Agent_C": "Agent_C", "Agent_D": "Agent_D", "END": END}
)

graph.add_edge(START, "Agent_A")
graph.add_edge("Agent_B", "Agent_A")
graph.add_edge("Agent_C", "Agent_A")
graph.add_edge("Agent_D", "Agent_A")

graph = graph.compile()
