from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader

text = """from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

class Review(TypedDict):
    summary: Annotated[str, 'A brief summary of the review']
    sentiment: Annotated[str, "Return sentiment of the review either negative, positive or neutral"]


structured_model = model.with_structured_output(Review)

print(result['summary'])
print(result['sentiment'])
"""
 

splitter =RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 100,
    chunk_overlap =0
)

result = splitter.split_text(text)
print(result[0])


