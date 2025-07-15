from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def generate_audience_profile(prompt: str):
    vectorstore = FAISS.load_local("vectorstores/audience_insights", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(prompt, k=3) # infer the target audience based on 3 most relevant documents from the prompt
    
    context = "\n".join([doc.page_content for doc in docs])
    llm = ChatOpenAI(temperature=0.3)

    final_prompt = f"""
    You're an expert marketing strategist.
    Use this context to infer the ideal audience for this campaign:
    
    Prompt: "{prompt}"
    Context:
    {context}

    Output a JSON profile with:
    - Persona Name
    - Age Group
    - Interests
    - Online Platforms
    - Motivations
    - Values
    """
    return llm.invoke(final_prompt)
