import os
from serpapi import GoogleSearch
from dotenv import load_dotenv
load_dotenv()

def search_google(query, num_results=10):
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    return GoogleSearch(params).get_dict("organic_results", [])