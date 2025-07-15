import asyncio
from types import SimpleNamespace
from agent.tools.market_research import market_research_agent

def test_market_research_agent():
    # Simulate LangGraph state using SimpleNamespace
    state = SimpleNamespace(
        prompt="Launch a TikTok campaign for a biodegradable water bottle",
        log=[]
    )
    config = {}

    # Run the async agent logic synchronously via asyncio
    result = asyncio.run(market_research_agent(state, config))

    # Assert expected structure of result
    assert "market_insights" in result
    assert "log" in result
    assert isinstance(result["market_insights"], dict)
    assert isinstance(result["log"], list)

    # # Print for debugging
    print("Market Insights:", result["market_insights"])
    print("Log:", result["log"])
