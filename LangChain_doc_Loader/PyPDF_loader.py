from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

chat_model = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen3-4B-Instruct-2507',
    task = 'text-generation',
    temperature=1.5
)

model = ChatHuggingFace(llm = chat_model)

prompt1 = PromptTemplate(
    template = ' On a Scale of 10 rate the poem \n {topic}',
    input_variables=['topic']
)

loader = PyPDFLoader(r'LangChain_doc_Loader/dl-curriculum.pdf')

docs = loader.load()

print(docs[0].page_content)

print(docs[0].metadata)