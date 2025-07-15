import streamlit as st
import time
import random
import json
from langchain_core.messages import HumanMessage
from agent.graph import graph  # Import the compiled graph

# --- Streamlit User Interface ---

st.set_page_config(page_title="AI Marketing Campaign Planner", layout="wide")

st.title("🤖 AI Marketing Campaign Planner")
st.write("Welcome to the 24-Hour Hackathon Edition! Let's plan a marketing campaign.")

# --- Sidebar for User Input ---

with st.sidebar:
    st.header("Campaign Brief")

    # New Feature: Dynamic and selectable example prompts.
    # Users can refresh to see different examples and choose one to use.
    example_prompts = [
        "Plan a digital marketing campaign for a new artisanal yuzu craft soda. The launch is in Kyoto, targeting both tourists and young local professionals. Highlight its unique flavor and local sourcing.",
        "Devise a launch strategy for a new mobile app that uses AI to create personalized travel itineraries. The target audience is millennial and Gen Z backpackers. Focus on social media and influencer collaborations.",
        "Create a content marketing plan for a B2B SaaS company that provides project management tools for remote teams. The goal is to increase free trial sign-ups. Focus on blog posts, case studies, and webinars.",
        "Outline a promotional campaign for a new line of sustainable, vegan leather handbags. The brand targets environmentally conscious consumers aged 25-45. Emphasize ethical production and high-quality materials."
    ]

    # Initialize the prompt index in session state if it doesn't exist.
    if 'prompt_index' not in st.session_state:
        st.session_state.prompt_index = 0

    # Display the current example prompt.
    st.info(f"**Example Prompt:**\n\n{example_prompts[st.session_state.prompt_index]}")

    # Create two columns for the buttons for a cleaner layout.
    col1, col2 = st.columns(2)

    with col1:
        # Refresh button: cycles to a new random prompt.
        if st.button("Refresh Prompt"):
            # Select a new random index.
            new_index = random.randint(0, len(example_prompts) - 1)
            st.session_state.prompt_index = new_index
            st.rerun()

    with col2:
        # Submit button: populates the text area with the current example.
        if st.button("Use This Prompt"):
            st.session_state.campaign_brief = example_prompts[st.session_state.prompt_index]

    campaign_brief = st.text_area(
        "Enter your product description, target audience, and goals:", 
        height=200,
        key="campaign_brief" 
    )
    generate_button = st.button("Generate Plan")

# --- Main Display Area ---

# If a plan was generated in a previous run, display it from session state
if 'plan_result' in st.session_state:
    result = st.session_state.plan_result
    st.success("Campaign Plan Generated!")
    with st.chat_message("ai", avatar="🤖"):
        st.markdown(result['final_summary'])
    st.markdown("---")
    st.subheader("Reasoning & Decision Log")
    for log_entry in result['log_trail']:
        with st.expander(f"**Agent: {log_entry['agent_name']}**"):
            st.markdown(f"**Decision:** {log_entry['decision']}")
            st.markdown("**Reasoning:**")
            st.markdown(f"```{json.dumps(log_entry['reasoning'], indent=2)}```")


if generate_button:
    if not campaign_brief:
        st.warning("Please enter a campaign brief in the sidebar.")
    else:
        # --- Streaming Execution and Real-Time Updates ---
        st.success("Campaign Plan Generation Started!")
        
        # Placeholders for the final summary and the log
        summary_placeholder = st.empty()
        log_container = st.container()
        log_container.markdown("---")
        log_container.subheader("Reasoning & Decision Log")

        with st.spinner("Our AI agents are collaborating... Please wait."):
            inputs = {"messages": [HumanMessage(content=campaign_brief)]}
            log_trail = []
            final_summary = ""

            # Use graph.stream() to get real-time updates
            for chunk in graph.stream(inputs, stream_mode="values"):
                # The 'chunk' contains the output of the last node that ran
                last_key = list(chunk.keys())[-1]
                
                if last_key == "Agent_A":
                    log_container.info("🧠 Supervisor is deciding the next step...")
                elif last_key == "Agent_B":
                    log_container.info("📈 Market Research Agent is analyzing trends...")
                    data = json.loads(chunk[last_key].get("data_B", "{}"))
                    log_entry = {
                        "agent_name": "Market Research Agent",
                        "decision": "Market analysis complete.",
                        "reasoning": data
                    }
                    log_trail.append(log_entry)
                elif last_key == "Agent_C":
                    log_container.info("👥 Audience Agent is profiling the target audience...")
                    data = json.loads(chunk[last_key].get("data_C", "{}"))
                    log_entry = {
                        "agent_name": "Audience Agent",
                        "decision": "Audience profiling complete.",
                        "reasoning": data
                    }
                    log_trail.append(log_entry)
                elif last_key == "Agent_D":
                    log_container.info("📝 Content Strategy Agent is creating content ideas...")
                    data = json.loads(chunk[last_key].get("data_D", "{}"))
                    log_entry = {
                        "agent_name": "Content Strategy Agent",
                        "decision": "Content strategy generation complete.",
                        "reasoning": data
                    }
                    log_trail.append(log_entry)

                # Extract final summary from the messages
                if "messages" in chunk:
                    final_summary = chunk["messages"][-1].content
        
        # --- Display Final Results and save to session state ---
        st.success("Campaign Plan Generated!")
        
        # Display Final Summary in a chat-like format
        with summary_placeholder.container():
             with st.chat_message("ai", avatar="🤖"):
                st.markdown(final_summary)

        # Display the detailed log trail in collapsible expanders
        for log_entry in log_trail:
            with log_container.expander(f"**Agent: {log_entry['agent_name']}**"):
                st.markdown(f"**Decision:** {log_entry['decision']}")
                st.markdown("**Reasoning:**")
                # Use st.json to pretty-print the dictionary
                st.json(log_entry['reasoning'])

        # Store the complete result in session state for persistence
        st.session_state.plan_result = {
            "final_summary": final_summary,
            "log_trail": log_trail
        }
        st.rerun() # Rerun to clean up placeholders and show final static view 