import os
from urllib import response
from serpapi import GoogleSearch
from dotenv import load_dotenv
load_dotenv()

def search_google_articles(query, num_results=10):
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    response = GoogleSearch(params).get_dict()
    results = response.get("organic_results", [])
    return results