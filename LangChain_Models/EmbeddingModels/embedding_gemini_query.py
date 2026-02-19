from langchain_google_genai import GoogleGenerativeAIEmbeddings  # FIXED IMPORT
from dotenv import load_dotenv
import os

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", 
    output_dimensionality=32
)

vectors = embedding.embed_query("Delhi is the capital of India")

print(str(vectors))