from __future__ import annotations
import json
import uuid
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum

class DecisionType(str, Enum):
    """Types of decisions that can be made in the reasoning process."""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"
    ANALYTICAL = "analytical"

class ConfidenceLevel(str, Enum):
    """Confidence levels for decisions and reasoning."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class ReasoningStage(str, Enum):
    """Stages of the reasoning process."""
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    DECISION = "decision"
    VALIDATION = "validation"

# ===== NEW COUNTERFACTUAL GENERATION MODELS =====

class CounterfactualRequest(BaseModel):
    """Request for generating counterfactuals for a specific agent."""
    agent_type: str = Field(..., description="Type of agent (market_research, audience_analysis, etc.)")
    user_prompt: str = Field(..., description="Original user campaign brief")
    current_analysis: Dict[str, Any] = Field(default_factory=dict, description="Current agent's analysis results")
    context_variables: Dict[str, Any] = Field(default_factory=dict, description="Extracted context variables")

class GeneratedCounterfactual(BaseModel):
    """A dynamically generated counterfactual scenario."""
    scenario_description: str = Field(..., description="What-if scenario description")
    likelihood: float = Field(..., ge=0.0, le=1.0, description="Probability this could occur")
    projected_outcome: str = Field(..., description="Expected outcome if this scenario occurred")
    impact_assessment: str = Field(..., description="How this would change the strategy")
    context_relevance: float = Field(..., ge=0.0, le=1.0, description="How relevant to user's specific context")

class CounterfactualResponse(BaseModel):
    """Response containing generated counterfactuals for an agent."""
    agent_type: str = Field(..., description="Agent these counterfactuals are for")
    counterfactuals: List[GeneratedCounterfactual] = Field(..., description="Generated counterfactual scenarios")
    context_summary: str = Field(..., description="Summary of key context factors considered")

# ===== EXISTING MODELS CONTINUE =====

class PacketHeader(BaseModel):
    """Header information for a reasoning packet."""
    packet_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="Identifier of the agent creating this packet")
    agent_name: str = Field(..., description="Human-readable name of the agent")
    timestamp: datetime = Field(default_factory=datetime.now)
    reasoning_stage: ReasoningStage = Field(..., description="Current stage of reasoning")
    parent_packet_id: Optional[str] = Field(None, description="ID of parent packet if this is a follow-up")
    session_id: str = Field(..., description="Session identifier for tracking related packets")

class CoreDecision(BaseModel):
    """Core decision information and rationale."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_type: DecisionType = Field(..., description="Type of decision being made")
    decision_statement: str = Field(..., description="Clear statement of what was decided")
    primary_objective: str = Field(..., description="Main goal this decision serves")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence in this decision")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Numerical confidence score")
    key_factors: List[str] = Field(..., description="Key factors that influenced this decision")
    assumptions: List[str] = Field(..., description="Assumptions underlying this decision")
    constraints: List[str] = Field(..., description="Constraints that limited options")
    expected_outcome: str = Field(..., description="Expected result of this decision")

class CounterfactualScenario(BaseModel):
    """Alternative scenario and its projected outcome."""
    scenario_description: str = Field(..., description="Description of the alternative scenario")
    likelihood: float = Field(..., ge=0.0, le=1.0, description="Likelihood this scenario could have occurred")
    projected_outcome: str = Field(..., description="Expected outcome if this scenario had occurred")
    impact_assessment: str = Field(..., description="Assessment of how this would change results")
    lessons_learned: Optional[str] = Field(None, description="What we learn from considering this scenario")

class EvidenceItem(BaseModel):
    """Piece of evidence supporting or challenging a decision."""
    evidence_type: Literal["data", "research", "expert_opinion", "precedent", "assumption"] = Field(...)
    source: str = Field(..., description="Source of this evidence")
    content: str = Field(..., description="The actual evidence content")
    reliability: float = Field(..., ge=0.0, le=1.0, description="Reliability score of this evidence")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance to the decision")
    supports_decision: bool = Field(..., description="Whether this evidence supports the decision")

class NarrativeLayer(BaseModel):
    """Narrative explanation of the reasoning process."""
    executive_summary: str = Field(..., description="High-level summary of the reasoning")
    detailed_rationale: str = Field(..., description="Detailed explanation of the reasoning process")
    decision_context: str = Field(..., description="Context in which this decision was made")
    stakeholder_considerations: List[str] = Field(..., description="How different stakeholders are affected")
    risk_assessment: str = Field(..., description="Assessment of risks associated with this decision")
    success_metrics: List[str] = Field(..., description="How success will be measured")
    implementation_notes: str = Field(..., description="Notes on how to implement this decision")

class CausalRelationship(BaseModel):
    """Causal relationship between factors."""
    cause: str = Field(..., description="The causal factor")
    effect: str = Field(..., description="The resulting effect")
    strength: float = Field(..., ge=0.0, le=1.0, description="Strength of the causal relationship")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this relationship")
    mediating_factors: List[str] = Field(default_factory=list, description="Factors that mediate this relationship")

class CausalLayer(BaseModel):
    """Causal reasoning and relationships."""
    root_causes: List[str] = Field(..., description="Root causes that led to this decision point")
    causal_chain: List[CausalRelationship] = Field(..., description="Chain of causal relationships")
    feedback_loops: List[str] = Field(..., description="Identified feedback loops in the system")
    unintended_consequences: List[str] = Field(..., description="Potential unintended consequences")
    systemic_effects: str = Field(..., description="How this decision affects the broader system")

class RobustnessTest(BaseModel):
    """Test of decision robustness under different conditions."""
    test_name: str = Field(..., description="Name of the robustness test")
    test_description: str = Field(..., description="What this test evaluates")
    scenario: str = Field(..., description="Scenario being tested")
    decision_holds: bool = Field(..., description="Whether the decision holds under this scenario")
    adaptation_required: Optional[str] = Field(None, description="Adaptations needed if scenario occurs")
    severity_if_fails: Literal["low", "medium", "high", "critical"] = Field(...)

class RobustnessLayer(BaseModel):
    """Robustness analysis of the decision."""
    sensitivity_analysis: str = Field(..., description="How sensitive the decision is to changes")
    scenario_testing: List[RobustnessTest] = Field(..., description="Results of scenario testing")
    counterfactuals: List[CounterfactualScenario] = Field(..., description="Alternative scenarios considered")
    stress_testing: str = Field(..., description="How the decision performs under stress")
    adaptability_assessment: str = Field(..., description="How adaptable the decision is to change")
    failure_modes: List[str] = Field(..., description="Ways this decision could fail")
    mitigation_strategies: List[str] = Field(..., description="Strategies to mitigate risks")

class AgentInteraction(BaseModel):
    """Record of interaction with other agents."""
    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="ID of the other agent")
    interaction_type: Literal["consultation", "handoff", "collaboration", "feedback"] = Field(...)
    interaction_summary: str = Field(..., description="Summary of the interaction")
    timestamp: datetime = Field(default_factory=datetime.now)

class ReasoningPacket(BaseModel):
    """Complete reasoning packet containing all layers of analysis."""
    header: PacketHeader = Field(..., description="Packet header information")
    core_decision: CoreDecision = Field(..., description="Core decision and rationale")
    narrative_layer: NarrativeLayer = Field(..., description="Narrative explanation")
    causal_layer: CausalLayer = Field(..., description="Causal reasoning")
    robustness_layer: RobustnessLayer = Field(..., description="Robustness analysis")
    evidence_base: List[EvidenceItem] = Field(..., description="Supporting evidence")
    agent_interactions: List[AgentInteraction] = Field(default_factory=list, description="Interactions with other agents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ReasoningPacket':
        """Create from JSON string."""
        return cls.model_validate_json(json_str)

# Simple output models for campaign strategy (separate from reasoning)
class CampaignStrategy(BaseModel):
    """Final campaign strategy output (separate from reasoning)."""
    strategy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    campaign_title: str = Field(..., description="Title of the campaign")
    executive_summary: str = Field(..., description="High-level campaign summary")
    target_audience_summary: str = Field(..., description="Summary of target audience")
    key_messages: List[str] = Field(..., description="Key marketing messages")
    recommended_channels: List[str] = Field(..., description="Recommended marketing channels")
    content_themes: List[str] = Field(..., description="Main content themes")
    success_metrics: List[str] = Field(..., description="How to measure success")
    timeline_overview: str = Field(..., description="High-level timeline")
    budget_considerations: str = Field(..., description="Budget and resource considerations")
    competitive_advantages: List[str] = Field(..., description="Key competitive advantages to highlight")

# Placeholder models for agent outputs
class MarketResearchSummary(BaseModel):
    """Summary of market research analysis."""
    trends: List[str] = Field(default_factory=list)
    competitors: List[str] = Field(default_factory=list)
    insights_summary: str = Field(default="Market analysis completed")

class AudienceInsightsSummary(BaseModel):
    """Summary of audience insights."""
    personas: List[str] = Field(default_factory=list)
    preferred_channels: List[str] = Field(default_factory=list)
    segmentation_strategy: str = Field(default="Audience segmentation completed")

class ContentStrategySummary(BaseModel):
    """Summary of content strategy."""
    content_pieces: List[str] = Field(default_factory=list)
    content_types: List[str] = Field(default_factory=list)
    strategy_overview: str = Field(default="Content strategy completed")

# Enhanced state management
class StateNDA(BaseModel):
    """Enhanced state management with reasoning tracking."""
    messages: List[Any] = Field(default_factory=list)
    reasoning_packets: List[ReasoningPacket] = Field(default_factory=list)
    campaign_strategy: Optional[CampaignStrategy] = Field(None)
    market_research: Optional[MarketResearchSummary] = Field(None)
    audience_insights: Optional[AudienceInsightsSummary] = Field(None)
    content_strategy: Optional[ContentStrategySummary] = Field(None)
    log_trail: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # NEW: Counterfactual support
    context_variables: Dict[str, Any] = Field(default_factory=dict, description="Extracted context variables")
    generated_counterfactuals: Dict[str, List[tuple]] = Field(default_factory=dict, description="Per-agent counterfactuals")
    
    # Legacy compatibility
    data_A: Optional[str] = Field(None)
    data_B: Optional[str] = Field(None)
    data_C: Optional[str] = Field(None)
    data_D: Optional[str] = Field(None)
    
    class Config:
        arbitrary_types_allowed = True 