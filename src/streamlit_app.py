import streamlit as st
import time

# --- Placeholder Backend Function ---

def run_hackathon_planner(brief: str):
    """
    This is a placeholder for your actual LangGraph backend.
    It simulates a call to the planner and returns a structured
    dictionary with a summary and a detailed log trail.

    In a real application, you would replace this function's body
    with a call to your LangGraph application. For example:
    
    from your_langgraph_module import your_graph
    
    def run_hackathon_planner(brief: str):
        inputs = {"brief": brief}
        # Assuming your LangGraph returns a dictionary with 'final_summary'
        # and 'log_trail' in its final state.
        result = your_graph.invoke(inputs)
        return result
    """
    # Simulate a network delay
    time.sleep(2)

    # Return a dummy dictionary matching the required structure
    return {
        "final_summary": f"This is the final campaign plan for: '{brief[:50]}...'. "
                         "We will focus on a multi-platform digital strategy, "
                         "targeting young professionals with a mix of engaging "
                         "video content and interactive social media campaigns.",
        "log_trail": [
            {
                "agent_name": "Market Research Agent",
                "decision": "Identified target audience as young professionals (25-35) on Instagram and TikTok.",
                "reasoning": "This demographic shows the highest engagement rates for similar product launches. "
                             "Counter-argument for targeting a broader audience was considered but rejected due to "
                             "budget constraints and the need for a high-impact initial launch."
            },
            {
                "agent_name": "Content Strategy Agent",
                "decision": "Develop a series of short, humorous video ads and a user-generated content contest.",
                "reasoning": "Video content is king on the identified platforms. A UGC contest will foster community "
                             "and generate authentic, low-cost marketing material. Static image ads were considered "
                             "less effective for this audience."
            },
            {
                "agent_name": "Ad Placement Agent",
                "decision": "Allocate 70% of the ad budget to Instagram Reels and 30% to TikTok.",
                "reasoning": "While TikTok has high engagement, Instagram's ad platform offers more precise targeting "
                             "options, which is crucial for maximizing ROI in the initial phase of the campaign."
            }
        ]
    }

# --- Streamlit User Interface ---

st.set_page_config(page_title="AI Marketing Campaign Planner", layout="wide")

st.title("🤖 AI Marketing Campaign Planner")
st.write("Welcome to the 24-Hour Hackathon Edition! Let's plan a marketing campaign.")

# --- Sidebar for User Input ---

with st.sidebar:
    st.header("Campaign Brief")
    campaign_brief = st.text_area(
        "Enter your product description, target audience, and goals:", 
        height=200
    )
    generate_button = st.button("Generate Plan")

# --- Main Display Area ---

if generate_button:
    if not campaign_brief:
        st.warning("Please enter a campaign brief in the sidebar.")
    else:
        with st.spinner("Our AI agents are collaborating... Please wait."):
            # Call the placeholder backend function
            plan_result = run_hackathon_planner(campaign_brief)
            # Store the result in session state
            st.session_state.plan_result = plan_result

# Check if a plan exists in the session state and display it
if 'plan_result' in st.session_state:
    result = st.session_state.plan_result
    
    st.success("Campaign Plan Generated!")

    # Display Final Summary in a chat-like format
    with st.chat_message("ai", avatar="🤖"):
        st.markdown(result['final_summary'])

    st.markdown("---")
    st.subheader("Reasoning & Decision Log")
    st.write("Here is the step-by-step log from the AI agent collaboration:")

    # Display the detailed log trail in collapsible expanders
    for log_entry in result['log_trail']:
        with st.expander(f"**Agent: {log_entry['agent_name']}**"):
            st.markdown(f"**Decision:** {log_entry['decision']}")
            st.markdown("**Reasoning:**")
            st.markdown(log_entry['reasoning']) 