from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from dotenv import load_dotenv
#from langchain_chroma import Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
load_dotenv()

 
embedding = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-large-en-v1.5",
    task="feature-extraction"
)

doc1 = Document(
    page_content="MS Dhoni is the legendary captain of Chennai Super Kings, known for his 'Captain Cool' demeanor and finishing abilities.",
    metadata={"player": "MS Dhoni", "role": "Wicketkeeper-Batsman", "team": "Chennai Super Kings"}
)

doc2 = Document(
    page_content="Ruturaj Gaikwad is a prolific opening batsman who has become a cornerstone of the CSK batting lineup with his elegant strokeplay.",
    metadata={"player": "Ruturaj Gaikwad", "role": "Batsman", "team": "Chennai Super Kings"}
)

doc3 = Document(
    page_content="Matheesha Pathirana, often called 'Baby Malinga', is a death-over specialist known for his unique slinging action and yorkers.",
    metadata={"player": "Matheesha Pathirana", "role": "Bowler", "team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Shivam Dube is a powerful left-handed batsman who has transformed into a spin-hitting monster for CSK in the middle overs.",
    metadata={"player": "Shivam Dube", "role": "All-rounder", "team": "Chennai Super Kings"}
)

doc5 = Document(
    page_content="Deepak Chahar is a swing bowling specialist who often provides crucial breakthroughs for CSK during the Powerplay overs.",
    metadata={"player": "Deepak Chahar", "role": "Bowler", "team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = DocArrayInMemorySearch.from_documents(
    docs, 
    embedding
)

vector_store.add_documents(docs)

result = vector_store.similarity_search(
    query='Who among these are Wicket-keeper',
    k = 1
)

result2 = result = vector_store.similarity_search_with_score(
    query='Who among these are Wicket-keeper',
    k = 1
)

