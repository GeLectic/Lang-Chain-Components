from langchain_google_genai import GoogleGenerativeAIEmbeddings  # FIXED IMPORT
from dotenv import load_dotenv
import os

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", 
    output_dimensionality=32
)
documents = [
    "Delhi is captial of india",
    "Bhopal is captial of MP",
    "Virat Kohli is Cricketer"
]


vectors = embedding.embed_documents(documents)

print(str(vectors))