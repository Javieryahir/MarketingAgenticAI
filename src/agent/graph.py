
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, TypedDict
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
import os
import uuid
from typing import Annotated, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Literal, Optional
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

# Cargar variables desde el archivo .env
load_dotenv()

# Acceder a las variables de entorno
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")

class SupervisorOutput(BaseModel):
    next_node: Literal["Agent_B", "Agent_C", "Agent_D", "END"]

class MarketResearchOutput(BaseModel):
    trends: list[str]
    competitors: list[str]
    insights_summary: str
    recommended_positioning: Optional[str] = None

class PersonaProfile(BaseModel):
    target_demographics: str
    psychographics: str
    behavior_and_habits: str
    challenges_or_pain_points: str
    effective_messaging_strategy: str
    summary_paragraph: str

class AudienceAgentOutput(BaseModel):
    personas: List[PersonaProfile]
    segmentation_strategy: str
    preferred_channels: List[str]

class ContentItem(BaseModel):
    description: str                        
    content_type: str                       
    campaign_theme: Optional[str] = None     
    upload_date: str                        
    upload_hour: str

class ExpandedContentStrategyOutput(BaseModel):
    items: list[ContentItem]  # Lista de piezas de contenido específicas



class StateNDA(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    data_A: Optional[str]
    data_B: Optional[str]
    data_C: Optional[str]
    data_D: Optional[str]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: StateNDA, config: RunnableConfig):
        state = {**state}
        result = self.runnable.invoke(state)
        return {"messages": result}


#SupervisorAgent
llm_a = ChatOpenAI(model="gpt-4o-mini")
llm_a_with_tools = llm_a.bind_tools([])
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are the Supervisor Agent responsible for coordinating a team of specialized agents to build a complete and effective marketing strategy.

        Your role is to manage the flow of the process, deciding which agent should be called next based on the current progress. Each agent performs a unique step in the campaign development pipeline.

        ### Available Agents:
        - Agent_B → Market Research
        - Agent_C → Audience Analysis
        - Agent_D → Content Strategy

### Recommended Workflow:
1. Start with Agent_B to understand market trends and competitors.
2. Then call Agent_C to define the target audience.
3. Next, activate Agent_D to develop the core messaging and content formats.
4. Finally, use Agent_E to create a publishing calendar with optimal times and channels.


### Your Task:
Analyze the previous messages and determine the most logical next step. Avoid repeating agents that have already been called. Once all steps are completed, return `"END"`.
        """),
    ("placeholder", "{messages}")
])
structured_chain: Runnable = prompt | llm_a.with_structured_output(SupervisorOutput)
text_chain_a: Runnable = prompt | llm_a

def Agent_A(state: StateNDA):
    state = {**state}

    structured_result = structured_chain.invoke(state)
    text_result = text_chain_a.invoke(state)

    return {
        "data_A": json.dumps(structured_result.dict(), ensure_ascii=False),
        "messages": text_result
    }

#MarketResearchAgent
llm_b = ChatOpenAI(model="gpt-4o-mini")
llm_b_with_tools = llm_b.bind_tools([])
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Market Research Agent specializing in gathering strategic insights for marketing campaigns.

Your task is to analyze the current market landscape for a given product or service based on the input conversation history. You should extract key trends, identify relevant competitors, summarize your insights, and suggest a positioning strategy.

### Your response should include:
- **trends**: A list of current and emerging trends relevant to the product or industry.
- **competitors**: A list of key competitors in the market, including a brief note on their strategies or positioning.
- **insights_summary**: A clear and concise summary of your overall findings.
- **recommended_positioning**: A suggestion for how the product or brand should be positioned in the market (optional but preferred).

Make your analysis clear, relevant, and focused on actionable insights that can inform the rest of the marketing strategy.
        """),
    ("placeholder", "{messages}")
])
structured_chain: Runnable = prompt | llm_b_with_tools.with_structured_output(MarketResearchOutput)
text_chain_b: Runnable = prompt | llm_b_with_tools

def Agent_B(state: StateNDA):
    state = {**state}

    structured_result = structured_chain.invoke(state)
    text_result = text_chain_b.invoke(state)

    return {
        "data_B": json.dumps(structured_result.dict(), ensure_ascii=False),
        "messages": text_result
    }
#changes
#AudienceAgent = Agent C
llm_c = ChatOpenAI(model="gpt-4o-mini")
llm_c_with_tools = llm_c.bind_tools([])
prompt = ChatPromptTemplate.from_messages([
    ("system", """
            You are an expert Audience Research Agent working for a marketing intelligence team.

Your task is to produce a structured audience profile for a given product, brand, or campaign idea. Your response must follow **this exact structure**, which will be used to populate a `AudienceAgentOutput` object.

Return a JSON object containing:

- `personas`: a list of 1–3 detailed persona profiles. Each persona must include:
    - `target_demographics`: Describe the persona’s age, gender, income level, location, education, etc.
    - `psychographics`: Interests, lifestyle, personality traits, values, or beliefs.
    - `behavior_and_habits`: Online behavior, purchase patterns, content preferences, typical platforms.
    - `challenges_or_pain_points`: What frustrates them, what they need solved, what problems they face.
    - `effective_messaging_strategy`: What type of tone, format, or message resonates with them.
    - `summary_paragraph`: A short fictional paragraph describing a person who represents this audience segment.
- `segmentation_strategy`: Describe how the audience could be segmented (e.g., by age, goals, habits, life stage, mindset).
- `preferred_channels`: A list of 2–5 channels/platforms where the campaign should focus (e.g., Instagram, TikTok, podcasts, email).
        """),
    ("placeholder", "{messages}")
])
structured_chain: Runnable = prompt | llm_c_with_tools.with_structured_output(AudienceAgentOutput)
text_chain_c: Runnable = prompt | llm_c_with_tools

def Agent_C(state: StateNDA):
    state = {**state}

    structured_result = structured_chain.invoke(state)
    text_result = text_chain_c.invoke(state)

    return {
        "data_B": json.dumps(structured_result.dict(), ensure_ascii=False),
        "messages": text_result
    }

#ContentStrategyAgent
llm_d = ChatOpenAI(model="gpt-4o-mini")
llm_d_with_tools = llm_d.bind_tools([])
prompt = ChatPromptTemplate.from_messages([
    ("system", """
     You are a Content Strategy Agent responsible for generating a list of content items for a marketing campaign.

Use the following contextual data to guide your strategy:

### 📊 Market Research Insights
{data_B}

### 🧠 Audience Profile
{data_C}

---

🎯 Your task is to create a structured list of content pieces, each including:

1. **description**: A brief but specific idea for the content (e.g., "How to save $500/month with simple hacks").
2. **content_type**: The format of the content (e.g., "video", "blog", "ad", "infographic", "podcast").
3. **campaign_theme** *(optional)*: A unifying theme or slogan if applicable (e.g., "Finance made fun").
4. **upload_date**: Suggested date for publishing the content (format: "YYYY-MM-DD").
5. **upload_hour**: Suggested time for publishing (24h format, e.g., "18:00").

🎯 Important:
- Tailor the content to the audience's demographics, interests, behaviors, and pain points.
- Leverage relevant insights from market research to align content with trends and opportunities.
- Vary the content types and schedule logically across the timeline.
        """),
    ("placeholder", "{messages}")
])
structured_chain: Runnable = prompt | llm_d_with_tools.with_structured_output(ExpandedContentStrategyOutput)
text_chain_d: Runnable = prompt | llm_d_with_tools

def Agent_D(state: StateNDA):
    state = {**state}

    structured_result = structured_chain.invoke(state)
    text_result = text_chain_d.invoke(state)

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