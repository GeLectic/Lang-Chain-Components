from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# embeddings
embedding = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-large-en-v1.5",
    task="feature-extraction"
)

# documents
documents = [
    Document(page_content="LangChain helps developers build LLM apps"),
    Document(page_content="FAISS is a vector database by Meta"),
    Document(page_content="Embeddings convert text into vectors"),
    Document(page_content="HuggingFace provides embedding models"),
    Document(page_content="MMR helps you get diverse results as well as relevant"),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone and more")
]

# create FAISS vector store
vectorStore = FAISS.from_documents(
    documents,
    embedding
)

retriever = vectorStore.as_retriever(
    search_type = "mmr"
    ,search_kwargs={"k":3, "lambda_mult": 0.5})

query = "What is LangChain"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n---Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")