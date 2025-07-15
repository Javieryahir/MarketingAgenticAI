from pytrends.request import TrendReq
import re
from typing import Dict, Any

def extract_keywords(prompt: str) -> list[str]:
    words = re.findall(r'\w+', prompt.lower())
    common_stopwords = {'the', 'is', 'in', 'for', 'a', 'an', 'and', 'to', 'of'}
    return [word for word in words if word not in common_stopwords and len(word) > 3]

async def market_research_agent(state, config: Dict[str, Any]) -> Dict[str, Any]:
    prompt = state.prompt
    keywords = extract_keywords(prompt)
    
    pytrends = TrendReq(hl='en-US', tz=360)

    trends_data = {}
    for keyword in keywords[:5]:  # We limit the top 5 keywords 
        try:
            pytrends.build_payload([keyword], cat=0, timeframe='now 7-d', geo='', gprop='')
            data = pytrends.interest_over_time()
            if not data.empty:
                trends_data[keyword] = data[keyword].to_dict()
        except Exception as e:
            trends_data[keyword] = {"error": str(e)}

    return {
        "market_insights": trends_data,
        "log": (state.log or []) + [f"MarketResearchAgent: Found trends for {keywords}."]
    }
