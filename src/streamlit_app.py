import streamlit as st
import time
import random
import json
from langchain_core.messages import HumanMessage
from agent.graph import graph, ReasoningPacket  # Import the graph and new data structure

# --- Streamlit User Interface ---

st.set_page_config(page_title="AI Marketing Campaign Planner", layout="wide")

st.title("🤖 AI Marketing Campaign Planner")
st.write(
    "Welcome to the Hackathon Edition! This app uses a multi-agent system "
    "to generate a marketing plan, with each agent's reasoning captured "
    "in a structured 'Reasoning Packet'."
)

# --- Sidebar for User Input ---

with st.sidebar:
    st.header("Campaign Brief")
    example_prompts = [
        "Plan a digital marketing campaign for a new artisanal yuzu craft soda. The launch is in Kyoto, targeting both tourists and young local professionals. Highlight its unique flavor and local sourcing.",
        "Devise a launch strategy for a new mobile app that uses AI to create personalized travel itineraries. The target audience is millennial and Gen Z backpackers. Focus on social media and influencer collaborations.",
        "Create a content marketing plan for a B2B SaaS company that provides project management tools for remote teams. The goal is to increase free trial sign-ups. Focus on blog posts, case studies, and webinars.",
    ]
    if 'prompt_index' not in st.session_state:
        st.session_state.prompt_index = 0
    st.info(f"**Example Prompt:**\n\n{example_prompts[st.session_state.prompt_index]}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Prompt"):
            new_index = random.randint(0, len(example_prompts) - 1)
            st.session_state.prompt_index = new_index
            st.rerun()
    with col2:
        if st.button("Use This Prompt"):
            st.session_state.campaign_brief = example_prompts[st.session_state.prompt_index]
    campaign_brief = st.text_area(
        "Enter your product description, target audience, and goals:",
        height=200,
        key="campaign_brief"
    )
    generate_button = st.button("Generate Plan")

# --- Main Display Area ---

def display_results(result_data):
    """Helper function to display the final results."""
    st.success("Campaign Plan Generated!")
    with st.chat_message("ai", avatar="🤖"):
        st.markdown(result_data.get('final_summary', "Processing complete."))
    
    st.markdown("---")
    st.subheader("Cognitive Narrative Reasoning Log")
    
    for packet_dict in result_data.get('log_trail', []):
        packet = ReasoningPacket.model_validate(packet_dict)
        header = packet.packetHeader
        with st.expander(f"**{header.agentID}** ({header.decisionEpoch}) - {packet.coreDecision.decisionStatement}"):
            st.markdown(f"**Rationalization:** {packet.narrativeLayer.rationalization}")
            st.markdown(f"**Confidence Score:** {packet.coreDecision.confidenceScore}")
            
            st.markdown("---")
            st.markdown("##### Core Decision Payload")
            st.json(packet.coreDecision.decisionPayload)

            st.markdown("---")
            st.markdown("##### Causal Layer")
            st.write("**Counterfactuals:**")
            st.json(packet.causalLayer.counterfactuals)


# Display previous results if they exist in session state
if 'plan_result' in st.session_state:
    display_results(st.session_state.plan_result)

if generate_button and 'plan_result' not in st.session_state:
    if not campaign_brief:
        st.warning("Please enter a campaign brief in the sidebar.")
    else:
        st.success("Campaign Plan Generation Started!")
        
        summary_placeholder = st.empty()
        log_container = st.container()
        
        with st.spinner("Our AI agents are collaborating..."):
            inputs = {"messages": [HumanMessage(content=campaign_brief)]}
            final_summary = "Processing complete."
            
            # Stream the graph execution
            for chunk in graph.stream(inputs, {"recursion_limit": 200}, stream_mode="values"):
                if "log_trail" in chunk:
                    log_container.empty()
                    with log_container:
                        st.subheader("Live Reasoning Log")
                        for packet in chunk['log_trail']:
                            header = packet.packetHeader
                            st.info(f"**{header.agentID}**: {packet.coreDecision.decisionStatement}")
                
                if "messages" in chunk:
                    final_summary = chunk["messages"][-1].content
        
        # After streaming, get the final state to build the full result
        final_state = graph.invoke(inputs, {"recursion_limit": 200})
        
        # Convert Pydantic models to dictionaries for session state
        log_trail_dicts = [packet.model_dump() for packet in final_state['log_trail']]

        st.session_state.plan_result = {
            "final_summary": final_summary,
            "log_trail": log_trail_dicts
        }
        st.rerun() 