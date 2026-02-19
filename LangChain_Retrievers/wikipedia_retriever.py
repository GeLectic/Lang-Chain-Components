from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from dotenv import load_dotenv
#from langchain_chroma import Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from langchain_community.retrievers import WikipediaRetriever
load_dotenv()

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "India-Pakistan relations China"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n---Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")