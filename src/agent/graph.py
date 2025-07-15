
from __future__ import annotations
import json
import os
from typing import Any, Dict, TypedDict, List
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
import uuid
from typing import Annotated, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool  
from langgraph.graph import StateGraph, END
from datetime import datetime
import re # Added for context extraction

# Import CNR framework
try:
    from .cnr_models import (
        ReasoningPacket, PacketHeader, CoreDecision, NarrativeLayer, 
        CausalLayer, RobustnessLayer, EvidenceItem, CounterfactualScenario,
        RobustnessTest, CausalRelationship, AgentInteraction,
        DecisionType, ConfidenceLevel, ReasoningStage,
        CampaignStrategy, MarketResearchSummary, AudienceInsightsSummary, 
        ContentStrategySummary, CounterfactualRequest, GeneratedCounterfactual,
        CounterfactualResponse
    )
except ImportError:
    from cnr_models import (
        ReasoningPacket, PacketHeader, CoreDecision, NarrativeLayer, 
        CausalLayer, RobustnessLayer, EvidenceItem, CounterfactualScenario,
        RobustnessTest, CausalRelationship, AgentInteraction,
        DecisionType, ConfidenceLevel, ReasoningStage,
        CampaignStrategy, MarketResearchSummary, AudienceInsightsSummary, 
        ContentStrategySummary, CounterfactualRequest, GeneratedCounterfactual,
        CounterfactualResponse
    )

# Load environment variables
load_dotenv()

# Environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")

# Legacy output models for backwards compatibility
class SupervisorOutput(BaseModel):
    next_node: Literal["Agent_CF", "Agent_B", "Agent_C", "Agent_D", "Agent_E", "END"]

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
    items: list[ContentItem]

# Enhanced state with CNR support
class StateNDA(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    reasoning_packets: List[str]  # JSON strings of ReasoningPackets
    campaign_strategy: Optional[str]  # JSON string of CampaignStrategy
    session_id: str
    log_trail: List[Dict[str, Any]]
    
    # NEW: Counterfactual support
    context_variables: Dict[str, Any]  # Extracted context variables
    generated_counterfactuals: Dict[str, List[tuple]]  # Per-agent counterfactuals
    
    # Legacy compatibility
    data_A: Optional[str]
    data_B: Optional[str] 
    data_C: Optional[str]
    data_D: Optional[str]

# ===== CONTEXT EXTRACTOR =====

class ContextExtractor:
    """Extracts key variables from user prompt for counterfactual generation."""
    
    @staticmethod
    def extract_context(user_prompt: str) -> Dict[str, Any]:
        """Extract key context variables from user prompt."""
        context = {
            "product_type": None,
            "target_location": None,
            "target_audience": [],
            "budget_mentioned": None,
            "timeline_mentioned": None,
            "channels_mentioned": [],
            "business_type": None,  # B2B vs B2C
            "industry": None,
            "campaign_goals": []
        }
        
        # Extract product/service type
        product_patterns = [
            r"(?:for|of)\s+(?:a\s+)?([^,.]+?)\s+(?:that|which|targeting)",
            r"marketing campaign for\s+([^,.]+)",
            r"plan.*for\s+([^,.]+)"
        ]
        for pattern in product_patterns:
            match = re.search(pattern, user_prompt, re.IGNORECASE)
            if match:
                context["product_type"] = match.group(1).strip()
                break
        
        # Extract location
        location_patterns = [
            r"in\s+([A-Z][a-zA-Z\s]+?)(?:,|\s+targeting|\.|$)",
            r"launch.*in\s+([A-Z][a-zA-Z\s]+)",
            r"market.*in\s+([A-Z][a-zA-Z\s]+)"
        ]
        for pattern in location_patterns:
            match = re.search(pattern, user_prompt, re.IGNORECASE)
            if match:
                context["target_location"] = match.group(1).strip()
                break
        
        # Extract audience
        audience_keywords = ["targeting", "target", "audience", "customers", "users"]
        for keyword in audience_keywords:
            pattern = fr"{keyword}[^.]*?([^,.]*(?:professional|tourist|millennial|gen z|consumer|business|student|parent)[^,.]*)"
            matches = re.findall(pattern, user_prompt, re.IGNORECASE)
            context["target_audience"].extend([m.strip() for m in matches])
        
        # Extract business type
        if re.search(r"b2b|business.to.business|businesses|enterprises", user_prompt, re.IGNORECASE):
            context["business_type"] = "B2B"
        elif re.search(r"b2c|business.to.consumer|consumers|customers", user_prompt, re.IGNORECASE):
            context["business_type"] = "B2C"
        
        # Extract channels
        channel_keywords = ["social media", "digital", "content marketing", "influencer", "email", "seo", "paid ads", "video", "blog"]
        for keyword in channel_keywords:
            if keyword.lower() in user_prompt.lower():
                context["channels_mentioned"].append(keyword)
        
        # Extract timeline
        timeline_patterns = [
            r"(\d+)\s+(?:month|week|day)s?",
            r"(q[1-4]|quarter)",
            r"(launch|go.live|start).*(?:in|by)\s+([^,.]+)"
        ]
        for pattern in timeline_patterns:
            match = re.search(pattern, user_prompt, re.IGNORECASE)
            if match:
                context["timeline_mentioned"] = match.group(0)
                break
        
        return context

# ===== COUNTERFACTUAL GENERATOR =====

class CounterfactualGenerator:
    """Generates contextual counterfactuals for different agent types."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.context_extractor = ContextExtractor()
    
    def generate_for_agent(self, agent_type: str, user_prompt: str, context_variables: Dict[str, Any]) -> List[tuple]:
        """Generate counterfactuals for a specific agent type."""
        
        # Get agent-specific prompt
        prompt = self._get_agent_prompt(agent_type)
        
        try:
            # Generate counterfactuals using LLM
            chain = prompt | self.llm.with_structured_output(CounterfactualResponse)
            
            response = chain.invoke({
                "user_prompt": user_prompt,
                "context_variables": json.dumps(context_variables, indent=2),
                "agent_type": agent_type
            })
            
            # Convert to tuple format for existing system compatibility
            cf_tuples = []
            for cf in response.counterfactuals:
                cf_tuples.append((
                    cf.scenario_description,
                    cf.likelihood,
                    cf.projected_outcome,
                    cf.impact_assessment
                ))
            
            return cf_tuples
            
        except Exception as e:
            # Fallback to generic counterfactuals if generation fails
            return self._get_fallback_counterfactuals(agent_type)
    
    def _get_agent_prompt(self, agent_type: str) -> ChatPromptTemplate:
        """Get agent-specific prompt for counterfactual generation."""
        
        prompts = {
            "supervisor": ChatPromptTemplate.from_messages([
                ("system", """
You are a Counterfactual Generator for the Supervisor Agent in a marketing campaign planning system.

Generate 2-3 workflow and coordination counterfactuals that are specific to this campaign context.

Consider:
- What if the workflow order was different for THIS specific campaign?
- What if certain agents were skipped for THIS product/market?
- What if resource constraints affected agent coordination?

Context Variables: {context_variables}
User Prompt: {user_prompt}

Generate realistic scenarios with likelihood scores (0.0-1.0) and specific impact assessments.
Focus on workflow decisions that would be relevant to THIS specific campaign.

Respond with exactly this JSON structure:
{{
    "agent_type": "{agent_type}",
    "counterfactuals": [
        {{
            "scenario_description": "If we...",
            "likelihood": 0.3,
            "projected_outcome": "Would result in...",
            "impact_assessment": "Impact would be...",
            "context_relevance": 0.8
        }}
    ],
    "context_summary": "Key context factors considered"
}}
"""),
                ("user", "Generate counterfactuals for {agent_type} agent")
            ]),
            
            "market_research": ChatPromptTemplate.from_messages([
                ("system", """
You are a Counterfactual Generator for the Market Research Agent.

Generate 3-4 market-specific counterfactuals based on the user's campaign context.

Consider the specific product, location, and market mentioned in the user prompt:
- What if market conditions were different in THIS specific location?
- What if THIS specific product faced different competitive pressures?
- What if market trends shifted for THIS industry/product category?
- What if research scope/budget was different for THIS campaign?

Context Variables: {context_variables}
User Prompt: {user_prompt}

Make scenarios specific to the product, market, and location mentioned.
Use realistic likelihood scores and detailed impact assessments.

Respond with exactly this JSON structure:
{{
    "agent_type": "{agent_type}",
    "counterfactuals": [
        {{
            "scenario_description": "If market...",
            "likelihood": 0.4,
            "projected_outcome": "Would require...",
            "impact_assessment": "Would change...",
            "context_relevance": 0.9
        }}
    ],
    "context_summary": "Market-specific factors analyzed"
}}
"""),
                ("user", "Generate counterfactuals for {agent_type} agent")
            ]),
            
            "audience_analysis": ChatPromptTemplate.from_messages([
                ("system", """
You are a Counterfactual Generator for the Audience Analysis Agent.

Generate 3-4 audience-specific counterfactuals based on the user's target audience and context.

Consider the specific audience segments mentioned in the user prompt:
- What if the target audience had different characteristics than assumed?
- What if audience preferences in THIS location were different?
- What if budget/channels limited how we could reach THIS specific audience?
- What if the audience was more/less receptive to THIS specific product?

Context Variables: {context_variables}
User Prompt: {user_prompt}

Make scenarios specific to the mentioned target audience, location, and product.
Consider cultural, demographic, and behavioral variations that could affect THIS campaign.

Respond with exactly this JSON structure:
{{
    "agent_type": "{agent_type}",
    "counterfactuals": [
        {{
            "scenario_description": "If audience...",
            "likelihood": 0.4,
            "projected_outcome": "Would need...",
            "impact_assessment": "Would require...",
            "context_relevance": 0.8
        }}
    ],
    "context_summary": "Audience-specific considerations"
}}
"""),
                ("user", "Generate counterfactuals for {agent_type} agent")
            ]),
            
            "content_strategy": ChatPromptTemplate.from_messages([
                ("system", """
You are a Counterfactual Generator for the Content Strategy Agent.

Generate 3-4 content-specific counterfactuals based on the user's campaign context.

Consider the specific product, audience, and channels mentioned:
- What if content production constraints were different for THIS campaign?
- What if THIS specific audience preferred different content formats?
- What if regulatory/cultural factors in THIS location affected content strategy?
- What if seasonal factors affected content timing for THIS product?

Context Variables: {context_variables}
User Prompt: {user_prompt}

Make scenarios specific to the product type, target audience, and location.
Consider practical content creation and distribution challenges.

Respond with exactly this JSON structure:
{{
    "agent_type": "{agent_type}",
    "counterfactuals": [
        {{
            "scenario_description": "If content...",
            "likelihood": 0.3,
            "projected_outcome": "Would adapt...",
            "impact_assessment": "Could maintain...",
            "context_relevance": 0.7
        }}
    ],
    "context_summary": "Content-specific factors"
}}
"""),
                ("user", "Generate counterfactuals for {agent_type} agent")
            ]),
            
            "campaign_generator": ChatPromptTemplate.from_messages([
                ("system", """
You are a Counterfactual Generator for the Campaign Strategy Generator.

Generate 3-4 high-level strategic counterfactuals that consider the overall campaign context.

Consider the complete campaign picture:
- What if overall market conditions changed for THIS specific context?
- What if budget/timeline constraints were different for THIS campaign?
- What if competitive responses were different in THIS market?
- What if success metrics needed to be different for THIS product/audience?

Context Variables: {context_variables}
User Prompt: {user_prompt}

Make scenarios that consider the integrated strategy implications.
Focus on campaign-level decisions that would affect overall success.

Respond with exactly this JSON structure:
{{
    "agent_type": "{agent_type}",
    "counterfactuals": [
        {{
            "scenario_description": "If strategy...",
            "likelihood": 0.3,
            "projected_outcome": "Would adapt...",
            "impact_assessment": "Demonstrates...",
            "context_relevance": 0.8
        }}
    ],
    "context_summary": "Strategic considerations"
}}
"""),
                ("user", "Generate counterfactuals for {agent_type} agent")
            ])
        }
        
        return prompts.get(agent_type, self._get_generic_prompt())
    
    def _get_generic_prompt(self) -> ChatPromptTemplate:
        """Generic fallback prompt."""
        return ChatPromptTemplate.from_messages([
            ("system", """
Generate 3 relevant counterfactual scenarios based on the user's campaign context.

Context Variables: {context_variables}
User Prompt: {user_prompt}

Respond with exactly this JSON structure:
{{
    "agent_type": "{agent_type}",
    "counterfactuals": [
        {{
            "scenario_description": "If...",
            "likelihood": 0.3,
            "projected_outcome": "Would...",
            "impact_assessment": "Impact...",
            "context_relevance": 0.6
        }}
    ],
    "context_summary": "General considerations"
}}
"""),
            ("user", "Generate counterfactuals for {agent_type} agent")
        ])
    
    def _get_fallback_counterfactuals(self, agent_type: str) -> List[tuple]:
        """Provide fallback counterfactuals if generation fails."""
        fallbacks = {
            "supervisor": [
                ("If workflow order was different", 0.3, "Different information flow", "Medium impact on coordination"),
                ("If agent specialization was bypassed", 0.2, "Reduced analysis quality", "High impact on strategy")
            ],
            "market_research": [
                ("If market conditions changed significantly", 0.4, "Strategy adaptation needed", "High impact on positioning"),
                ("If competitive landscape shifted", 0.3, "Repositioning required", "Medium impact on strategy")
            ],
            "audience_analysis": [
                ("If target audience preferences differed", 0.4, "Channel and messaging adjustments", "Medium-high impact"),
                ("If audience accessibility was limited", 0.3, "Alternative targeting needed", "Medium impact on reach")
            ],
            "content_strategy": [
                ("If production constraints were tighter", 0.3, "Format optimization needed", "Medium impact on output"),
                ("If content preferences varied", 0.4, "Strategy refinement required", "Medium impact on engagement")
            ],
            "campaign_generator": [
                ("If market conditions evolved", 0.3, "Strategic flexibility needed", "Medium impact on adaptation"),
                ("If resource constraints changed", 0.4, "Scope adjustment required", "Medium impact on execution")
            ]
        }
        
        return fallbacks.get(agent_type, [
            ("If conditions changed", 0.3, "Adaptation needed", "Medium impact")
        ])

# ===== ASSEMBLER FUNCTIONS FOR CNR REASONING =====

def create_reasoning_packet(
    agent_id: str,
    agent_name: str,
    decision_statement: str,
    decision_type: DecisionType,
    detailed_rationale: str,
    key_factors: List[str],
    evidence_items: List[tuple],  # (type, source, content, reliability, relevance, supports)
    counterfactuals: List[tuple],  # (description, likelihood, outcome, impact)
    session_id: str,
    confidence_score: float = 0.85,
    constraints: List[str] = None,
    assumptions: List[str] = None,
    **kwargs
) -> ReasoningPacket:
    """Assembler function to create a complete ReasoningPacket programmatically."""
    
    if constraints is None:
        constraints = []
    if assumptions is None:
        assumptions = []
    
    # Create header
    header = PacketHeader(
        agent_id=agent_id,
        agent_name=agent_name,
        reasoning_stage=ReasoningStage.DECISION,
        session_id=session_id
    )
    
    # Create core decision
    core_decision = CoreDecision(
        decision_type=decision_type,
        decision_statement=decision_statement,
        primary_objective=kwargs.get('primary_objective', f"Complete {agent_name} analysis"),
        confidence_level=ConfidenceLevel.HIGH if confidence_score > 0.8 else ConfidenceLevel.MEDIUM,
        confidence_score=confidence_score,
        key_factors=key_factors,
        assumptions=assumptions,
        constraints=constraints,
        expected_outcome=kwargs.get('expected_outcome', "Informed decision-making for campaign strategy")
    )
    
    # Create evidence base
    evidence_base = []
    for evidence_tuple in evidence_items:
        evidence_base.append(EvidenceItem(
            evidence_type=evidence_tuple[0],
            source=evidence_tuple[1],
            content=evidence_tuple[2],
            reliability=evidence_tuple[3],
            relevance=evidence_tuple[4],
            supports_decision=evidence_tuple[5]
        ))
    
    # Create counterfactuals
    counterfactual_scenarios = []
    for cf_tuple in counterfactuals:
        counterfactual_scenarios.append(CounterfactualScenario(
            scenario_description=cf_tuple[0],
            likelihood=cf_tuple[1],
            projected_outcome=cf_tuple[2],
            impact_assessment=cf_tuple[3]
        ))
    
    # Create narrative layer
    narrative_layer = NarrativeLayer(
        executive_summary=kwargs.get('executive_summary', f"{agent_name} completed analysis successfully"),
        detailed_rationale=detailed_rationale,
        decision_context=kwargs.get('decision_context', "Marketing campaign planning context"),
        stakeholder_considerations=kwargs.get('stakeholder_considerations', ["Marketing team", "Target customers", "Business stakeholders"]),
        risk_assessment=kwargs.get('risk_assessment', "Standard market risks considered"),
        success_metrics=kwargs.get('success_metrics', ["Campaign effectiveness", "Audience engagement", "Business objectives"]),
        implementation_notes=kwargs.get('implementation_notes', "Integrate findings into overall campaign strategy")
    )
    
    # Create causal layer
    causal_relationships = []
    for factor in key_factors[:3]:  # Use top factors for causal relationships
        causal_relationships.append(CausalRelationship(
            cause=factor,
            effect="Strategic direction informed",
            strength=0.8,
            confidence=0.75,
            mediating_factors=["Market conditions", "Resource availability"]
        ))
    
    causal_layer = CausalLayer(
        root_causes=kwargs.get('root_causes', ["Market dynamics", "Customer needs", "Business objectives"]),
        causal_chain=causal_relationships,
        feedback_loops=kwargs.get('feedback_loops', ["Customer feedback influences strategy", "Market response shapes positioning"]),
        unintended_consequences=kwargs.get('unintended_consequences', ["Potential competitor response", "Resource allocation changes"]),
        systemic_effects=kwargs.get('systemic_effects', "Influences overall campaign coherence and effectiveness")
    )
    
    # Create robustness layer
    robustness_tests = [
        RobustnessTest(
            test_name="Market volatility test",
            test_description="How well decision holds under market changes",
            scenario="High market volatility scenario",
            decision_holds=True,
            severity_if_fails="medium"
        ),
        RobustnessTest(
            test_name="Resource constraint test",
            test_description="Decision viability under resource limitations",
            scenario="Reduced budget scenario",
            decision_holds=True,
            adaptation_required="Scale down scope proportionally",
            severity_if_fails="low"
        )
    ]
    
    robustness_layer = RobustnessLayer(
        sensitivity_analysis=kwargs.get('sensitivity_analysis', "Decision moderately sensitive to major market changes"),
        scenario_testing=robustness_tests,
        counterfactuals=counterfactual_scenarios,
        stress_testing=kwargs.get('stress_testing', "Decision holds under normal stress conditions"),
        adaptability_assessment=kwargs.get('adaptability_assessment', "High adaptability to changing conditions"),
        failure_modes=kwargs.get('failure_modes', ["Data quality issues", "Market disruption", "Resource constraints"]),
        mitigation_strategies=kwargs.get('mitigation_strategies', ["Continuous monitoring", "Flexible implementation", "Backup scenarios"])
    )
    
    return ReasoningPacket(
        header=header,
        core_decision=core_decision,
        narrative_layer=narrative_layer,
        causal_layer=causal_layer,
        robustness_layer=robustness_layer,
        evidence_base=evidence_base,
        metadata=kwargs.get('metadata', {})
    )

# ===== AGENT IMPLEMENTATIONS WITH CNR =====

# Counterfactual Generator Agent (Agent_CF) - NEW
llm_cf = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # Lower temp for consistency

def Agent_CF(state: StateNDA):
    """Counterfactual Generator Agent - runs early to generate context-specific counterfactuals."""
    
    # Extract user prompt
    messages = state.get('messages', [])
    user_prompt = ""
    if messages:
        # Get the last human message
        for msg in reversed(messages):
            if hasattr(msg, 'content'):
                user_prompt = str(msg.content)
                break
    
    if not user_prompt:
        user_prompt = "Marketing campaign planning"
    
    # Initialize counterfactual generator
    cf_generator = CounterfactualGenerator()
    
    # Extract context variables
    context_variables = cf_generator.context_extractor.extract_context(user_prompt)
    
    # Generate counterfactuals for each agent type
    agent_types = ["supervisor", "market_research", "audience_analysis", "content_strategy", "campaign_generator"]
    generated_counterfactuals = {}
    total_generated = 0
    
    for agent_type in agent_types:
        try:
            cf_tuples = cf_generator.generate_for_agent(agent_type, user_prompt, context_variables)
            generated_counterfactuals[agent_type] = cf_tuples
            total_generated += len(cf_tuples)
        except Exception as e:
            # Use fallback counterfactuals
            fallback_cfs = cf_generator._get_fallback_counterfactuals(agent_type)
            generated_counterfactuals[agent_type] = fallback_cfs
            total_generated += len(fallback_cfs)
    
    # Create reasoning packet for the counterfactual generation process
    evidence_items = [
        ("research", "Context Extraction", f"Identified {len([v for v in context_variables.values() if v])} context variables", 0.9, 0.95, True),
        ("research", "Counterfactual Creation", f"Generated {total_generated} scenarios across {len(agent_types)} agent types", 0.85, 0.9, True),
        ("expert_opinion", "LLM Analysis", f"Context-aware counterfactuals for {context_variables.get('product_type', 'campaign')}", 0.8, 0.9, True)
    ]
    
    counterfactuals = [
        ("If context extraction missed key variables", 0.2, "Less relevant counterfactuals generated", "Medium impact on reasoning quality"),
        ("If LLM generated generic scenarios", 0.3, "Reduced contextual relevance", "High impact on counterfactual value")
    ]
    
    reasoning_packet = create_reasoning_packet(
        agent_id="Agent_CF",
        agent_name="Counterfactual Generator",
        decision_statement=f"Generated {total_generated} context-specific counterfactuals for all {len(agent_types)} agents",
        decision_type=DecisionType.ANALYTICAL,
        detailed_rationale=f"Analyzed user prompt for '{context_variables.get('product_type', 'campaign')}' in '{context_variables.get('target_location', 'unspecified location')}' targeting '{', '.join(context_variables.get('target_audience', ['general audience']))}' and generated contextually relevant counterfactuals for each specialist agent.",
        key_factors=[
            "Context variable extraction",
            "Domain-specific scenario generation", 
            "Likelihood assessment",
            "Impact analysis customization",
            "Agent-specific relevance"
        ],
        evidence_items=evidence_items,
        counterfactuals=counterfactuals,
        session_id=state.get('session_id', str(uuid.uuid4())),
        confidence_score=0.88,
        primary_objective="Generate contextually relevant counterfactuals for enhanced reasoning",
        decision_context="Pre-analysis counterfactual generation phase",
        success_metrics=["Context extraction accuracy", "Counterfactual relevance", "Agent-specific alignment"]
    )
    
    # Update state
    reasoning_packets = state.get('reasoning_packets', [])
    reasoning_packets.append(reasoning_packet.to_json())
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "Agent_CF",
        "action": "counterfactual_generation",
        "context_variables_extracted": len([v for v in context_variables.values() if v]),
        "total_counterfactuals_generated": total_generated,
        "agent_types_processed": len(agent_types),
        "reasoning_id": reasoning_packet.header.packet_id
    }
    
    log_trail = state.get('log_trail', [])
    log_trail.append(log_entry)
    
    return {
        "context_variables": context_variables,
        "generated_counterfactuals": generated_counterfactuals,
        "reasoning_packets": reasoning_packets,
        "log_trail": log_trail,
        "messages": f"Generated {total_generated} contextual counterfactuals based on campaign context: {context_variables.get('product_type', 'product')} in {context_variables.get('target_location', 'market')}"
    }

# Supervisor Agent (Agent_A) - CNR Enhanced
llm_a = ChatOpenAI(model="gpt-4o-mini")

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the Supervisor Agent for a marketing campaign planning system. Your role is to coordinate specialist agents and make routing decisions.

CRITICAL: You must analyze the current state and decide which agent should be called next. Consider:
- Agent_CF: Counterfactual Generator (generate context-specific counterfactuals) - MUST RUN FIRST
- Agent_B: Market Research (analyze trends, competitors, positioning)
- Agent_C: Audience Analysis (develop personas, segmentation, channels)  
- Agent_D: Content Strategy (create content calendar and strategic pieces)
- Agent_E: Campaign Generator (synthesize final campaign strategy)

Recommended workflow:
1. FIRST: Agent_CF to generate context-specific counterfactuals
2. Then Agent_B for market research
3. Then Agent_C for audience analysis
4. Then Agent_D for content strategy
5. Then Agent_E to generate final campaign
6. END when complete

Check the state to see if counterfactuals have been generated. If not, route to Agent_CF first.

Respond with your routing decision and reasoning for why this agent should be called next.
"""),
    ("placeholder", "{messages}")
])

def Agent_A(state: StateNDA):
    """Supervisor Agent with CNR reasoning."""
    
    # Get LLM decision
    structured_chain = supervisor_prompt | llm_a.with_structured_output(SupervisorOutput)
    text_chain = supervisor_prompt | llm_a
    
    structured_result = structured_chain.invoke(state)
    text_result = text_chain.invoke(state)
    
    # Extract routing decision
    next_node = structured_result.next_node
    
    # Create detailed CNR reasoning packet
    evidence_items = [
        ("expert_opinion", "LLM Analysis", f"Routing decision: {next_node}", 0.9, 0.95, True),
        ("precedent", "Workflow Standards", "Following established agent workflow", 0.95, 0.9, True)
    ]
    
    # Get dynamic counterfactuals instead of static ones
    counterfactuals = state.get('generated_counterfactuals', {}).get('supervisor', [
        # Fallback to generic if generation failed
        ("If workflow order was different", 0.3, "Different information flow", "Medium impact on coordination"),
        ("If agent specialization was bypassed", 0.2, "Reduced analysis quality", "High impact on strategy")
    ])
    
    reasoning_packet = create_reasoning_packet(
        agent_id="Agent_A",
        agent_name="Supervisor Agent", 
        decision_statement=f"Route workflow to {next_node}",
        decision_type=DecisionType.OPERATIONAL,
        detailed_rationale=f"Based on current workflow state and agent completion status, the optimal next step is to engage {next_node}. This maintains logical information flow and ensures each specialist contributes their expertise in the proper sequence.",
        key_factors=["Workflow sequence optimization", "Information dependency management", "Agent specialization alignment"],
        evidence_items=evidence_items,
        counterfactuals=counterfactuals,
        session_id=state.get('session_id', str(uuid.uuid4())),
        confidence_score=0.92,
        primary_objective="Maintain optimal agent workflow coordination",
        decision_context="Multi-agent marketing campaign planning system",
        success_metrics=["Workflow efficiency", "Information quality", "Agent coordination"]
    )
    
    # Update state
    reasoning_packets = state.get('reasoning_packets', [])
    reasoning_packets.append(reasoning_packet.to_json())
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "Agent_A",
        "action": "routing_decision",
        "decision": next_node,
        "reasoning_id": reasoning_packet.header.packet_id
    }
    
    log_trail = state.get('log_trail', [])
    log_trail.append(log_entry)
    
    return {
        "data_A": json.dumps(structured_result.dict(), ensure_ascii=False),
        "messages": text_result,
        "reasoning_packets": reasoning_packets,
        "log_trail": log_trail,
        "session_id": state.get('session_id', str(uuid.uuid4()))
    }

# Market Research Agent (Agent_B) - CNR Enhanced
llm_b = ChatOpenAI(model="gpt-4o-mini")

market_research_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Market Research Agent specializing in strategic market analysis for marketing campaigns.

Analyze the market landscape for the given product/service. Focus on:
- Current and emerging market trends
- Key competitors and their strategies
- Market opportunities and gaps
- Recommended positioning strategy

Provide clear, actionable insights that will inform audience analysis and content strategy.
"""),
    ("placeholder", "{messages}")
])

def Agent_B(state: StateNDA):
    """Market Research Agent with CNR reasoning."""
    
    # Get LLM analysis
    structured_chain = market_research_prompt | llm_b.with_structured_output(MarketResearchOutput)
    text_chain = market_research_prompt | llm_b
    
    structured_result = structured_chain.invoke(state)
    text_result = text_chain.invoke(state)
    
    # Create comprehensive CNR reasoning packet
    trends_count = len(structured_result.trends)
    competitors_count = len(structured_result.competitors)
    
    evidence_items = [
        ("research", "Market Analysis", f"Identified {trends_count} key trends", 0.85, 0.9, True),
        ("research", "Competitive Analysis", f"Analyzed {competitors_count} key competitors", 0.8, 0.85, True),
        ("expert_opinion", "LLM Analysis", structured_result.insights_summary, 0.9, 0.95, True)
    ]
    
    # Get dynamic counterfactuals instead of static ones
    counterfactuals = state.get('generated_counterfactuals', {}).get('market_research', [
        # Fallback to generic if generation failed
        ("If market conditions changed significantly", 0.4, "Strategy adaptation needed", "High impact on positioning"),
        ("If competitive landscape shifted", 0.3, "Repositioning required", "Medium impact on strategy")
    ])
    
    reasoning_packet = create_reasoning_packet(
        agent_id="Agent_B",
        agent_name="Market Research Agent",
        decision_statement=f"Market analysis complete with {trends_count} trends and {competitors_count} competitors identified",
        decision_type=DecisionType.ANALYTICAL,
        detailed_rationale=f"Conducted comprehensive market analysis identifying key trends, competitive landscape, and positioning opportunities. Analysis reveals {structured_result.insights_summary}. Recommended positioning: {structured_result.recommended_positioning}",
        key_factors=[
            f"Market trend analysis ({trends_count} trends identified)",
            f"Competitive landscape mapping ({competitors_count} competitors)",
            "Strategic positioning development",
            "Market opportunity identification"
        ],
        evidence_items=evidence_items,
        counterfactuals=counterfactuals,
        session_id=state.get('session_id', str(uuid.uuid4())),
        confidence_score=0.87,
        primary_objective="Provide foundational market insights for campaign strategy",
        expected_outcome="Informed audience targeting and content positioning",
        decision_context="Marketing campaign market research phase",
        stakeholder_considerations=["Marketing team", "Product team", "Business development"],
        success_metrics=["Market insight quality", "Competitive advantage identification", "Positioning clarity"]
    )
    
    # Update state
    reasoning_packets = state.get('reasoning_packets', [])
    reasoning_packets.append(reasoning_packet.to_json())
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "Agent_B", 
        "action": "market_analysis",
        "trends_identified": trends_count,
        "competitors_analyzed": competitors_count,
        "reasoning_id": reasoning_packet.header.packet_id
    }
    
    log_trail = state.get('log_trail', [])
    log_trail.append(log_entry)
    
    return {
        "data_B": json.dumps(structured_result.dict(), ensure_ascii=False),
        "messages": text_result,
        "reasoning_packets": reasoning_packets,
        "log_trail": log_trail
    }

# Audience Analysis Agent (Agent_C) - CNR Enhanced
llm_c = ChatOpenAI(model="gpt-4o-mini")

audience_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert Audience Research Agent for marketing intelligence.

Create detailed audience profiles including:
- 1-3 detailed personas with demographics, psychographics, behaviors
- Audience segmentation strategy
- Preferred communication channels

Structure your response for AudienceAgentOutput with complete persona profiles.
"""),
    ("placeholder", "{messages}")
])

def Agent_C(state: StateNDA):
    """Audience Analysis Agent with CNR reasoning."""
    
    # Get LLM analysis
    structured_chain = audience_prompt | llm_c.with_structured_output(AudienceAgentOutput)
    text_chain = audience_prompt | llm_c
    
    structured_result = structured_chain.invoke(state)
    text_result = text_chain.invoke(state)
    
    # Create CNR reasoning packet
    personas_count = len(structured_result.personas)
    channels_count = len(structured_result.preferred_channels)
    
    evidence_items = [
        ("research", "Audience Analysis", f"Developed {personas_count} detailed personas", 0.88, 0.92, True),
        ("research", "Channel Analysis", f"Identified {channels_count} optimal channels", 0.85, 0.9, True),
        ("expert_opinion", "Segmentation Strategy", structured_result.segmentation_strategy, 0.9, 0.95, True)
    ]
    
    # Get dynamic counterfactuals instead of static ones
    counterfactuals = state.get('generated_counterfactuals', {}).get('audience_analysis', [
        # Fallback to generic if generation failed
        ("If target audience preferences differed", 0.4, "Channel and messaging adjustments", "Medium-high impact"),
        ("If audience accessibility was limited", 0.3, "Alternative targeting needed", "Medium impact on reach")
    ])
    
    reasoning_packet = create_reasoning_packet(
        agent_id="Agent_C",
        agent_name="Audience Research Agent",
        decision_statement=f"Audience analysis complete with {personas_count} personas and {channels_count} channels",
        decision_type=DecisionType.STRATEGIC,
        detailed_rationale=f"Comprehensive audience analysis resulted in {personas_count} detailed personas covering target demographics, psychographics, and behavioral patterns. Segmentation strategy: {structured_result.segmentation_strategy}. Optimal channels identified: {', '.join(structured_result.preferred_channels)}",
        key_factors=[
            f"Persona development ({personas_count} personas)",
            "Psychographic profiling",
            "Behavioral pattern analysis", 
            f"Channel optimization ({channels_count} channels)",
            "Segmentation strategy design"
        ],
        evidence_items=evidence_items,
        counterfactuals=counterfactuals,
        session_id=state.get('session_id', str(uuid.uuid4())),
        confidence_score=0.91,
        primary_objective="Define target audience for precise campaign targeting",
        expected_outcome="Data-driven audience targeting and channel selection",
        decision_context="Marketing campaign audience research phase",
        success_metrics=["Persona accuracy", "Channel effectiveness", "Segmentation clarity"]
    )
    
    # Update state
    reasoning_packets = state.get('reasoning_packets', [])
    reasoning_packets.append(reasoning_packet.to_json())
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "Agent_C",
        "action": "audience_analysis", 
        "personas_created": personas_count,
        "channels_identified": channels_count,
        "reasoning_id": reasoning_packet.header.packet_id
    }
    
    log_trail = state.get('log_trail', [])
    log_trail.append(log_entry)
    
    return {
        "data_C": json.dumps(structured_result.dict(), ensure_ascii=False),
        "messages": text_result,
        "reasoning_packets": reasoning_packets,
        "log_trail": log_trail
    }

# Content Strategy Agent (Agent_D) - CNR Enhanced
llm_d = ChatOpenAI(model="gpt-4o-mini")

content_strategy_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Content Strategy Agent responsible for creating comprehensive content plans.

Develop a strategic content calendar including:
- Specific content pieces with descriptions
- Content types and formats
- Publishing dates and optimal timing
- Campaign themes and messaging

Consider audience preferences and market insights for optimal content strategy.
"""),
    ("placeholder", "{messages}")
])

def Agent_D(state: StateNDA):
    """Content Strategy Agent with CNR reasoning."""
    
    # Get LLM analysis
    structured_chain = content_strategy_prompt | llm_d.with_structured_output(ExpandedContentStrategyOutput)
    text_chain = content_strategy_prompt | llm_d
    
    structured_result = structured_chain.invoke(state)
    text_result = text_chain.invoke(state)
    
    # Create CNR reasoning packet
    content_count = len(structured_result.items)
    content_types = set(item.content_type for item in structured_result.items)
    
    evidence_items = [
        ("research", "Content Planning", f"Created {content_count} strategic content pieces", 0.87, 0.9, True),
        ("research", "Format Analysis", f"Diversified across {len(content_types)} content types", 0.85, 0.88, True),
        ("expert_opinion", "Strategic Alignment", "Content aligned with audience personas and market positioning", 0.9, 0.95, True)
    ]
    
    # Get dynamic counterfactuals instead of static ones
    counterfactuals = state.get('generated_counterfactuals', {}).get('content_strategy', [
        # Fallback to generic if generation failed
        ("If production constraints were tighter", 0.3, "Format optimization needed", "Medium impact on output"),
        ("If content preferences varied", 0.4, "Strategy refinement required", "Medium impact on engagement")
    ])
    
    reasoning_packet = create_reasoning_packet(
        agent_id="Agent_D",
        agent_name="Content Strategy Agent",
        decision_statement=f"Content strategy complete with {content_count} pieces across {len(content_types)} formats",
        decision_type=DecisionType.TACTICAL,
        detailed_rationale=f"Developed comprehensive content strategy with {content_count} strategic content pieces optimized for audience engagement and market positioning. Content spans {len(content_types)} different formats ({', '.join(content_types)}) to maximize reach and effectiveness across identified channels.",
        key_factors=[
            f"Content calendar development ({content_count} pieces)",
            f"Format diversification ({len(content_types)} types)",
            "Audience alignment optimization",
            "Timeline and scheduling strategy",
            "Campaign theme integration"
        ],
        evidence_items=evidence_items,
        counterfactuals=counterfactuals,
        session_id=state.get('session_id', str(uuid.uuid4())),
        confidence_score=0.89,
        primary_objective="Create actionable content strategy for campaign execution",
        expected_outcome="Strategic content calendar ready for implementation",
        decision_context="Marketing campaign content planning phase",
        success_metrics=["Content engagement potential", "Format effectiveness", "Timeline optimization"]
    )
    
    # Update state
    reasoning_packets = state.get('reasoning_packets', [])
    reasoning_packets.append(reasoning_packet.to_json())
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "Agent_D",
        "action": "content_strategy",
        "content_pieces": content_count,
        "content_types": len(content_types),
        "reasoning_id": reasoning_packet.header.packet_id
    }
    
    log_trail = state.get('log_trail', [])
    log_trail.append(log_entry)
    
    return {
        "data_D": json.dumps(structured_result.dict(), ensure_ascii=False),
        "messages": text_result,
        "reasoning_packets": reasoning_packets,
        "log_trail": log_trail
    }

# Campaign Strategy Generator (Agent_E) - NEW
llm_e = ChatOpenAI(model="gpt-4o-mini")

campaign_generator_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the Campaign Strategy Generator. Your role is to synthesize all previous agent analyses into a comprehensive, actionable marketing campaign strategy.

Based on the market research, audience analysis, and content strategy, create a cohesive campaign that includes:
- Campaign title and executive summary
- Key marketing messages
- Target audience summary
- Recommended channels and content themes
- Success metrics and timeline overview
- Budget considerations and competitive advantages

Make this a practical, implementable strategy that business stakeholders can execute.
"""),
    ("placeholder", "{messages}")
])

def Agent_E(state: StateNDA):
    """Campaign Strategy Generator with final campaign output."""
    
    # Get campaign strategy from LLM
    campaign_result = campaign_generator_prompt | llm_e
    campaign_text = campaign_result.invoke(state)
    
    # Parse previous agent data to create structured campaign strategy
    market_data = None
    audience_data = None  
    content_data = None
    
    try:
        if state.get('data_B'):
            market_data = json.loads(state['data_B'])
        if state.get('data_C'):
            audience_data = json.loads(state['data_C'])
        if state.get('data_D'):
            content_data = json.loads(state['data_D'])
    except:
        pass
    
    # Create structured campaign strategy
    campaign_strategy = CampaignStrategy(
        campaign_title="Strategic Marketing Campaign",
        executive_summary=str(campaign_text.content)[:500] + "..." if len(str(campaign_text.content)) > 500 else str(campaign_text.content),
        target_audience_summary=f"Target audience analysis complete with {len(audience_data.get('personas', []))} personas" if audience_data else "Comprehensive audience analysis completed",
        key_messages=[
            "Value proposition aligned with market positioning",
            "Customer-centric messaging strategy", 
            "Competitive differentiation focus"
        ],
        recommended_channels=audience_data.get('preferred_channels', ['Digital channels', 'Social media', 'Content marketing']) if audience_data else ['Multi-channel approach'],
        content_themes=[
            "Brand awareness and education",
            "Customer engagement and retention",
            "Conversion optimization"
        ],
        success_metrics=[
            "Campaign reach and impressions",
            "Engagement rates and interactions", 
            "Conversion rates and ROI",
            "Brand awareness metrics"
        ],
        timeline_overview="Phased approach: Research → Strategy → Content Creation → Launch → Optimization",
        budget_considerations="Budget allocation optimized across research, content creation, media spend, and performance monitoring",
        competitive_advantages=[
            "Data-driven audience targeting",
            "Comprehensive market positioning",
            "Integrated content strategy"
        ]
    )
    
    # Create reasoning packet for campaign generation
    evidence_items = [
        ("research", "Market Analysis", f"Market research with {len(market_data.get('trends', []))} trends" if market_data else "Market research completed", 0.9, 0.95, True),
        ("research", "Audience Analysis", f"Audience analysis with {len(audience_data.get('personas', []))} personas" if audience_data else "Audience analysis completed", 0.9, 0.95, True),
        ("research", "Content Strategy", f"Content strategy with {len(content_data.get('items', []))} pieces" if content_data else "Content strategy completed", 0.9, 0.95, True)
    ]
    
    # Get dynamic counterfactuals instead of static ones
    counterfactuals = state.get('generated_counterfactuals', {}).get('campaign_generator', [
        # Fallback to generic if generation failed
        ("If market conditions evolved", 0.3, "Strategic flexibility needed", "Medium impact on adaptation"),
        ("If resource constraints changed", 0.4, "Scope adjustment required", "Medium impact on execution")
    ])
    
    reasoning_packet = create_reasoning_packet(
        agent_id="Agent_E",
        agent_name="Campaign Strategy Generator",
        decision_statement="Comprehensive campaign strategy synthesized from all agent analyses",
        decision_type=DecisionType.STRATEGIC,
        detailed_rationale="Integrated insights from market research, audience analysis, and content strategy to create a cohesive, actionable marketing campaign. Strategy balances market opportunities with audience preferences and practical implementation considerations.",
        key_factors=[
            "Cross-agent insight integration",
            "Strategic coherence optimization",
            "Implementation feasibility", 
            "Stakeholder value alignment",
            "Performance measurement framework"
        ],
        evidence_items=evidence_items,
        counterfactuals=counterfactuals,
        session_id=state.get('session_id', str(uuid.uuid4())),
        confidence_score=0.93,
        primary_objective="Deliver comprehensive, actionable marketing campaign strategy",
        expected_outcome="Ready-to-implement campaign strategy with clear success metrics",
        decision_context="Final campaign strategy synthesis phase",
        success_metrics=["Strategy completeness", "Implementation readiness", "Stakeholder alignment"]
    )
    
    # Update state
    reasoning_packets = state.get('reasoning_packets', [])
    reasoning_packets.append(reasoning_packet.to_json())
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "Agent_E",
        "action": "campaign_generation",
        "strategy_id": campaign_strategy.strategy_id,
        "reasoning_id": reasoning_packet.header.packet_id
    }
    
    log_trail = state.get('log_trail', [])
    log_trail.append(log_entry)
    
    return {
        "messages": campaign_text,
        "campaign_strategy": campaign_strategy.model_dump_json(indent=2),
        "reasoning_packets": reasoning_packets,
        "log_trail": log_trail
    }

# ===== GRAPH SETUP =====

graph = StateGraph(StateNDA)

# Register all agent nodes
graph.add_node("Agent_A", Agent_A)  # Supervisor
graph.add_node("Agent_B", Agent_B)  # Market Research  
graph.add_node("Agent_C", Agent_C)  # Audience Analysis
graph.add_node("Agent_D", Agent_D)  # Content Strategy
graph.add_node("Agent_E", Agent_E)  # Campaign Generator
graph.add_node("Agent_CF", Agent_CF) # Counterfactual Generator

graph.set_entry_point("Agent_A")

# After each specialist agent finishes, return to supervisor
graph.add_edge("Agent_CF", "Agent_A")  # Counterfactual generator returns to supervisor
graph.add_edge("Agent_B", "Agent_A")
graph.add_edge("Agent_C", "Agent_A") 
graph.add_edge("Agent_D", "Agent_A")
graph.add_edge("Agent_E", END)  # Campaign generator goes to END

# Router function for supervisor decisions
def router(state):
    """Enhanced router with CNR logging."""
    data_A_str = state.get("data_A")
    if data_A_str:
        try:
            supervisor_data = json.loads(data_A_str)
            next_node = supervisor_data.get("next_node", "END")
            return next_node
        except:
            return "END"
    return "END"

graph.add_conditional_edges(
    "Agent_A",
    router,
    {
        "Agent_CF": "Agent_CF",
        "Agent_B": "Agent_B",
        "Agent_C": "Agent_C", 
        "Agent_D": "Agent_D",
        "Agent_E": "Agent_E",
        "END": END
    }
)

# Compile the graph
graph = graph.compile()
