from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('LangChain_doc_Loader\dl-curriculum.pdf')

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap =0
)

result = splitter.split_documents(docs)
print(result)


