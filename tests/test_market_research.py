import asyncio
from types import SimpleNamespace
from agent.market_research import market_research_agent

# Simulated State object (LangGraph would normally manage this)
state = SimpleNamespace(
    prompt="Launch a TikTok campaign for a biodegradable water bottle",
    log=[]
)

config = {}

# Run the async agent function using asyncio
async def run_test():
    result = await market_research_agent(state, config)
    print("📊 Market Insights:\n", result["market_insights"])
    print("\n📝 Log:\n", result["log"])

# Execute the test
if __name__ == "__main__":
    asyncio.run(run_test())
