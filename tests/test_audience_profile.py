from agent.tools.ingest_web2vector_store import ingest_and_store
from agent.tools.audience_profile_tool import generate_audience_profile

def test_generate_audience_profile(): # example test function to run the example for the audience_profile tool
    prompt = "Create a marketing strategy for college students who want healthy snacks"
    ingest_and_store(prompt)
    print("Ingested and stored data for:", prompt)
    profile = generate_audience_profile(prompt)

    assert "Persona Name" in profile.content  # basic sanity check
    assert isinstance(profile.content, str)
    print(profile.content) # return the generated audience profile


