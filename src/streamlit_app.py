import streamlit as st
import time
import random
import json
from langchain_core.messages import HumanMessage
try:
    from agent.graph import graph  # Import just the graph
    from agent.cnr_models import ReasoningPacket, CampaignStrategy
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))
    from graph import graph
    from cnr_models import ReasoningPacket, CampaignStrategy
from datetime import datetime

# --- Streamlit User Interface ---

st.set_page_config(page_title="AI Marketing Campaign Planner", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .persona-card {
        background: #fff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #764ba2;
    }
    .content-item {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .reasoning-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b35;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .evidence-item {
        background: #e8f4fd;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
    }
    .counterfactual-item {
        background: #fff3e0;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9800;
    }
    .causal-relationship {
        background: #f3e5f5;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #9c27b0;
    }
    .robustness-test {
        background: #e8f5e8;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #4caf50;
    }
    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .supervisor-badge { background: #e3f2fd; color: #1976d2; }
    .research-badge { background: #f3e5f5; color: #7b1fa2; }
    .audience-badge { background: #e8f5e8; color: #2e7d32; }
    .content-badge { background: #fff3e0; color: #f57c00; }
    .campaign-badge { background: #fce4ec; color: #c2185b; }
    .campaign-strategy-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Marketing Campaign Planner</h1>
    <p>CNR-Enhanced Multi-Agent AI System for Comprehensive Campaign Strategy</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("📋 Campaign Brief")
    
    # Example prompts
    example_prompts = [
        "Plan a digital marketing campaign for a new artisanal yuzu craft soda. The launch is in Kyoto, targeting both tourists and young local professionals. Highlight its unique flavor and local sourcing.",
        "Devise a launch strategy for a new mobile app that uses AI to create personalized travel itineraries. The target audience is millennial and Gen Z backpackers. Focus on social media and influencer collaborations.",
        "Create a content marketing plan for a B2B SaaS company that provides project management tools for remote teams. The goal is to increase free trial sign-ups. Focus on blog posts, case studies, and webinars.",
        "Design a marketing campaign for an eco-friendly fashion brand targeting environmentally conscious consumers aged 25-40. Focus on sustainability, ethical production, and style.",
        "Launch campaign for a new fitness app with AI personal trainer features. Target busy professionals who want convenient, personalized workouts."
    ]
    
    if 'prompt_index' not in st.session_state:
        st.session_state.prompt_index = 0
    
    with st.expander("💡 Example Prompts", expanded=False):
        st.write(f"**Current Example:**")
        st.info(example_prompts[st.session_state.prompt_index])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Example"):
                st.session_state.prompt_index = random.randint(0, len(example_prompts) - 1)
                st.rerun()
        with col2:
            if st.button("📝 Use This"):
                st.session_state.campaign_brief = example_prompts[st.session_state.prompt_index]
                st.rerun()
    
    campaign_brief = st.text_area(
        "Describe your product, target audience, and campaign goals:",
        height=150,
        key="campaign_brief",
        placeholder="Enter your campaign brief here..."
    )
    
    generate_button = st.button("🚀 Generate Campaign Plan", type="primary", use_container_width=True)
    
    if 'plan_result' in st.session_state:
        st.markdown("---")
        st.success("✅ Campaign Generated!")
        if st.button("🗑️ Clear Results", use_container_width=True):
            del st.session_state.plan_result
            st.rerun()

# --- Helper Functions for CNR Display ---

def display_reasoning_packet(packet_json: str, agent_icon: str = "🤖"):
    """Display a complete CNR reasoning packet."""
    try:
        packet = ReasoningPacket.from_json(packet_json)
        
        with st.expander(f"{agent_icon} **{packet.header.agent_name}** - {packet.core_decision.decision_statement}", expanded=False):
            
            # Header Information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{packet.core_decision.confidence_score:.1%}")
            with col2:
                st.metric("Decision Type", packet.core_decision.decision_type.value.title())
            with col3:
                st.metric("Reasoning Stage", packet.header.reasoning_stage.value.title())
            
            st.markdown("---")
            
            # Core Decision & Narrative
            st.markdown("### 💭 **Decision Rationale**")
            st.write(packet.narrative_layer.detailed_rationale)
            
            st.markdown("### 🎯 **Primary Objective**")
            st.info(packet.core_decision.primary_objective)
            
            # Key Factors
            st.markdown("### 🔑 **Key Factors Considered**")
            for factor in packet.core_decision.key_factors:
                st.write(f"• {factor}")
            
            # Evidence Base
            if packet.evidence_base:
                st.markdown("### 📊 **Evidence Base**")
                for evidence in packet.evidence_base:
                    reliability_color = "🟢" if evidence.reliability > 0.8 else "🟡" if evidence.reliability > 0.6 else "🔴"
                    support_indicator = "✅" if evidence.supports_decision else "❌"
                    
                    st.markdown(f"""
                    <div class="evidence-item">
                        <strong>{support_indicator} {evidence.evidence_type.title()}</strong> {reliability_color}<br/>
                        <strong>Source:</strong> {evidence.source}<br/>
                        <strong>Content:</strong> {evidence.content}<br/>
                        <small>Reliability: {evidence.reliability:.1%} | Relevance: {evidence.relevance:.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Counterfactual Analysis
            if packet.robustness_layer.counterfactuals:
                st.markdown("### 🤔 **Alternative Scenarios Considered**")
                for cf in packet.robustness_layer.counterfactuals:
                    likelihood_indicator = "🔴" if cf.likelihood > 0.7 else "🟡" if cf.likelihood > 0.4 else "🟢"
                    
                    st.markdown(f"""
                    <div class="counterfactual-item">
                        <strong>What if:</strong> {cf.scenario_description} {likelihood_indicator}<br/>
                        <strong>Projected outcome:</strong> {cf.projected_outcome}<br/>
                        <strong>Impact assessment:</strong> {cf.impact_assessment}<br/>
                        <small>Likelihood: {cf.likelihood:.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Causal Analysis
            if packet.causal_layer.causal_chain:
                st.markdown("### 🔗 **Causal Relationships**")
                for relationship in packet.causal_layer.causal_chain:
                    strength_indicator = "🔴" if relationship.strength > 0.8 else "🟡" if relationship.strength > 0.6 else "🟢"
                    
                    st.markdown(f"""
                    <div class="causal-relationship">
                        <strong>Cause:</strong> {relationship.cause}<br/>
                        <strong>Effect:</strong> {relationship.effect} {strength_indicator}<br/>
                        <strong>Mediating factors:</strong> {', '.join(relationship.mediating_factors)}<br/>
                        <small>Strength: {relationship.strength:.1%} | Confidence: {relationship.confidence:.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Robustness Testing
            if packet.robustness_layer.scenario_testing:
                st.markdown("### 🛡️ **Robustness Testing**")
                for test in packet.robustness_layer.scenario_testing:
                    test_result = "✅ PASSES" if test.decision_holds else "❌ FAILS"
                    severity_color = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(test.severity_if_fails, "⚪")
                    
                    st.markdown(f"""
                    <div class="robustness-test">
                        <strong>{test.test_name}:</strong> {test_result}<br/>
                        <strong>Scenario:</strong> {test.scenario}<br/>
                        <strong>Severity if fails:</strong> {test.severity_if_fails.title()} {severity_color}<br/>
                        {f"<strong>Adaptation required:</strong> {test.adaptation_required}<br/>" if test.adaptation_required else ""}
                        <small>{test.test_description}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Risk Assessment
            st.markdown("### ⚠️ **Risk Assessment**")
            st.warning(packet.narrative_layer.risk_assessment)
            
            # Success Metrics
            st.markdown("### 📈 **Success Metrics**")
            for metric in packet.narrative_layer.success_metrics:
                st.write(f"• {metric}")
            
            # Technical Details
            with st.expander("🔧 Technical Details", expanded=False):
                st.json({
                    "packet_id": packet.header.packet_id,
                    "timestamp": packet.header.timestamp.isoformat(),
                    "session_id": packet.header.session_id,
                    "assumptions": packet.core_decision.assumptions,
                    "constraints": packet.core_decision.constraints,
                    "systemic_effects": packet.causal_layer.systemic_effects,
                    "implementation_notes": packet.narrative_layer.implementation_notes
                })
    
    except Exception as e:
        st.error(f"Error displaying reasoning packet: {str(e)}")

def display_campaign_strategy(strategy_json: str):
    """Display the final campaign strategy."""
    try:
        strategy = CampaignStrategy.model_validate_json(strategy_json)
        
        st.markdown(f"""
        <div class="campaign-strategy-box">
            <h2>🎯 {strategy.campaign_title}</h2>
            <h4>📝 Executive Summary</h4>
            <p style="font-size: 1.1em; line-height: 1.6;">{strategy.executive_summary}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Campaign Components
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👥 **Target Audience**")
            st.info(strategy.target_audience_summary)
            
            st.markdown("### 💬 **Key Messages**")
            for message in strategy.key_messages:
                st.write(f"• {message}")
            
            st.markdown("### 📱 **Recommended Channels**")
            for channel in strategy.recommended_channels:
                st.write(f"• {channel}")
        
        with col2:
            st.markdown("### 🎨 **Content Themes**")
            for theme in strategy.content_themes:
                st.write(f"• {theme}")
            
            st.markdown("### 📊 **Success Metrics**")
            for metric in strategy.success_metrics:
                st.write(f"• {metric}")
            
            st.markdown("### 🏆 **Competitive Advantages**")
            for advantage in strategy.competitive_advantages:
                st.write(f"• {advantage}")
        
        # Timeline and Budget
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### ⏰ **Timeline Overview**")
            st.success(strategy.timeline_overview)
        
        with col4:
            st.markdown("### 💰 **Budget Considerations**")
            st.success(strategy.budget_considerations)
        
    except Exception as e:
        st.error(f"Error displaying campaign strategy: {str(e)}")

def display_execution_log(log_trail: list):
    """Display the execution log trail."""
    st.markdown("### 📋 **Execution Timeline**")
    
    for i, entry in enumerate(reversed(log_trail[-10:])):  # Show last 10 entries
        timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
        
        agent_icons = {
            "Agent_A": "🎯",
            "Agent_B": "📊", 
            "Agent_C": "👥",
            "Agent_D": "📝",
            "Agent_E": "🚀"
        }
        
        icon = agent_icons.get(entry['agent'], "🤖")
        action_desc = entry.get('action', 'unknown').replace('_', ' ').title()
        
        with st.container():
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin: 0.3rem 0; border-left: 3px solid #667eea;">
                <strong>{icon} {entry['agent']}</strong> - {action_desc}<br/>
                <small>🕒 {timestamp} | ID: {entry.get('reasoning_id', 'N/A')[:8]}...</small>
            </div>
            """, unsafe_allow_html=True)

def extract_campaign_data(state):
    """Extract structured data from the state for backwards compatibility."""
    market_data = None
    audience_data = None
    content_data = None
    
    # Parse data from each agent
    if state.get('data_B'):
        try:
            market_data = json.loads(state['data_B'])
        except:
            pass
    
    if state.get('data_C'):
        try:
            audience_data = json.loads(state['data_C'])
        except:
            pass
    
    if state.get('data_D'):
        try:
            content_data = json.loads(state['data_D'])
        except:
            pass
    
    return market_data, audience_data, content_data

def display_market_research(data):
    """Display market research in a structured format."""
    if not data:
        return
    
    st.subheader("📊 Market Research Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔥 Key Trends**")
        for trend in data.get('trends', []):
            st.markdown(f"• {trend}")
    
    with col2:
        st.markdown("**🏢 Key Competitors**")
        for competitor in data.get('competitors', []):
            st.markdown(f"• {competitor}")
    
    if data.get('insights_summary'):
        st.markdown("**💡 Strategic Insights**")
        st.info(data['insights_summary'])
    
    if data.get('recommended_positioning'):
        st.markdown("**🎯 Recommended Positioning**")
        st.success(data['recommended_positioning'])

def display_audience_analysis(data):
    """Display audience analysis with persona cards."""
    if not data:
        return
    
    st.subheader("👥 Audience Analysis")
    
    # Segmentation Strategy
    if data.get('segmentation_strategy'):
        st.markdown("**📋 Segmentation Strategy**")
        st.info(data['segmentation_strategy'])
    
    # Preferred Channels
    if data.get('preferred_channels'):
        st.markdown("**📱 Preferred Channels**")
        channels = " • ".join(data['preferred_channels'])
        st.markdown(f"**Primary Focus:** {channels}")
    
    # Personas
    personas = data.get('personas', [])
    if personas:
        st.markdown("**👤 Target Personas**")
        
        for i, persona in enumerate(personas, 1):
            with st.container():
                st.markdown(f"""
                <div class="persona-card">
                    <h4>🎭 Persona {i}: {persona.get('target_demographics', 'Unknown').split(',')[0]}</h4>
                    <p><strong>Demographics:</strong> {persona.get('target_demographics', 'N/A')}</p>
                    <p><strong>Psychographics:</strong> {persona.get('psychographics', 'N/A')}</p>
                    <p><strong>Behaviors:</strong> {persona.get('behavior_and_habits', 'N/A')}</p>
                    <p><strong>Pain Points:</strong> {persona.get('challenges_or_pain_points', 'N/A')}</p>
                    <p><strong>Messaging Strategy:</strong> {persona.get('effective_messaging_strategy', 'N/A')}</p>
                    <div style="background: #e8f5e8; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                        <strong>📝 Profile:</strong> {persona.get('summary_paragraph', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

def display_content_strategy(data):
    """Display content strategy with timeline."""
    if not data:
        return
    
    st.subheader("📝 Content Strategy")
    
    # Handle both 'items' and 'content_items' field names
    content_items = data.get('items', data.get('content_items', []))
    
    if content_items:
        # Content type distribution
        content_types = {}
        for item in content_items:
            ctype = item.get('content_type', 'Unknown')
            content_types[ctype] = content_types.get(ctype, 0) + 1
        
        st.markdown("**📊 Content Mix**")
        cols = st.columns(len(content_types))
        for i, (ctype, count) in enumerate(content_types.items()):
            with cols[i]:
                st.metric(ctype.title(), count)
        
        st.markdown("**📅 Content Calendar**")
        
        # Sort by date
        sorted_items = sorted(content_items, key=lambda x: x.get('upload_date', ''))
        
        for item in sorted_items:
            with st.container():
                st.markdown(f"""
                <div class="content-item">
                    <h5>📄 {item.get('description', 'Untitled Content')}</h5>
                    <p><strong>Type:</strong> {item.get('content_type', 'N/A')} | 
                       <strong>Date:</strong> {item.get('upload_date', 'N/A')} | 
                       <strong>Time:</strong> {item.get('upload_hour', 'N/A')}</p>
                    {f"<p><strong>Theme:</strong> {item.get('campaign_theme', 'N/A')}</p>" if item.get('campaign_theme') else ""}
                </div>
                """, unsafe_allow_html=True)

# --- Main Display Logic ---

# Display previous results if they exist
if 'plan_result' in st.session_state:
    result_data = st.session_state.plan_result
    
    # Main campaign plan display using tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Campaign Strategy", "🧠 CNR Reasoning", "📊 Component Analysis", "🔧 Technical Data"])
    
    with tab1:
        st.markdown("## 🎉 **Marketing Campaign Strategy**")
        
        # Display final campaign strategy if available
        if result_data.get('campaign_strategy'):
            display_campaign_strategy(result_data['campaign_strategy'])
        else:
            # Fallback to legacy display
            st.success("🎉 **Campaign Plan Generated Successfully!**")
            
            # Final summary
            if result_data.get('final_summary'):
                st.markdown("### 📝 Executive Summary")
                with st.container():
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745;">
                        {result_data['final_summary']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show execution timeline
        if result_data.get('log_trail'):
            display_execution_log(result_data['log_trail'])
    
    with tab2:
        st.markdown("## 🧠 **Cognitive Narrative Reasoning (CNR) Analysis**")
        st.markdown("*Detailed decision-making process and reasoning for each agent*")
        
        # Display all reasoning packets
        reasoning_packets = result_data.get('reasoning_packets', [])
        
        if reasoning_packets:
            st.info(f"📦 **{len(reasoning_packets)} reasoning packets** generated during campaign planning")
            
            agent_icons = {
                "Agent_A": "🎯",
                "Agent_B": "📊", 
                "Agent_C": "👥",
                "Agent_D": "📝",
                "Agent_E": "🚀"
            }
            
            for packet_json in reasoning_packets:
                try:
                    packet_preview = ReasoningPacket.from_json(packet_json)
                    icon = agent_icons.get(packet_preview.header.agent_id, "🤖")
                    display_reasoning_packet(packet_json, icon)
                except Exception as e:
                    st.error(f"Error parsing reasoning packet: {str(e)}")
        else:
            st.warning("No CNR reasoning packets found. The system may be running in legacy mode.")
    
    with tab3:
        st.markdown("## 📊 **Detailed Component Analysis**")
        
        # Extract structured data for backwards compatibility
        market_data, audience_data, content_data = extract_campaign_data(result_data)
        
        # Display structured campaign components
        if market_data:
            display_market_research(market_data)
            st.markdown("---")
        
        if audience_data:
            display_audience_analysis(audience_data)
            st.markdown("---")
            
        if content_data:
            display_content_strategy(content_data)
    
    with tab4:
        st.markdown("## 🔧 **Technical Data & Raw Output**")
        
        # Agent Messages
        st.subheader("💬 Agent Communication Log")
        if result_data.get('messages'):
            for i, msg in enumerate(result_data['messages']):
                with st.expander(f"Message {i+1} - {type(msg).__name__}", expanded=False):
                    st.write(f"**Content:** {msg.content if hasattr(msg, 'content') else str(msg)}")
                    if hasattr(msg, 'type'):
                        st.write(f"**Type:** {msg.type}")
        
        # Raw state data
        st.subheader("🗂️ Raw State Data")
        st.json(result_data)

# Generate new plan
if generate_button and 'plan_result' not in st.session_state:
    if not campaign_brief:
        st.warning("⚠️ Please enter a campaign brief in the sidebar.")
    else:
        st.success("🚀 **Campaign Generation Started!**")
        
        # Live updates section
        progress_bar = st.progress(0)
        status_text = st.empty()
        live_log = st.empty()
        
        with st.spinner("🤖 AI agents are collaborating with CNR reasoning..."):
            inputs = {
                "messages": [HumanMessage(content=campaign_brief)],
                "reasoning_packets": [],
                "session_id": f"session_{int(time.time())}",
                "log_trail": [],
                "campaign_strategy": None,
                "data_A": None,
                "data_B": None,
                "data_C": None,
                "data_D": None
            }
            
            # Stream the graph execution
            step_count = 0
            max_steps = 20  # Increased for new agent
            current_messages = []
            
            for chunk in graph.stream(inputs, {"recursion_limit": 200}, stream_mode="values"):
                step_count += 1
                progress = min(step_count / max_steps, 0.95)
                progress_bar.progress(progress)
                
                if "messages" in chunk:
                    current_messages = chunk['messages']
                    
                    # Show live reasoning updates
                    agent = "Processing"
                    action = "Working"
                    
                    if chunk.get('log_trail'):
                        latest_log = chunk['log_trail'][-1] if chunk['log_trail'] else {}
                        agent = latest_log.get('agent', 'Unknown')
                        action = latest_log.get('action', 'processing').replace('_', ' ').title()
                        
                        with live_log.container():
                            st.markdown(f"""
                            <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                                <strong>🔄 {agent}:</strong> {action}...<br/>
                                <small>Step {step_count}/{max_steps}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    status_text.text(f"Step {step_count}: {agent} - {action}")
        
        # Get final state
        progress_bar.progress(1.0)
        status_text.text("✅ Finalizing campaign strategy with CNR analysis...")
        
        final_state = graph.invoke(inputs, {"recursion_limit": 200})
        
        # Store results
        st.session_state.plan_result = final_state
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        live_log.empty()
        
        st.rerun() 