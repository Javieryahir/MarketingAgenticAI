from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agent.tools.serpapi_search import search_google_articles

def ingest_and_store(query):
    results = search_google_articles(query)
    docs = [Document(page_content=r['title'] + ". " + r.get('snippet', '')) for r in results]

    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings()) # store the information chunks in vector store
    vectorstore.save_local("vectorstores/audience_insights") # save the vector data locally 