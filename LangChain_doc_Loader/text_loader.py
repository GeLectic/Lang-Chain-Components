from langchain_community.document_loaders import TextLoader
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


loader = TextLoader('LangChain_doc_Loader/cricket.txt')

docs = loader.load()

parser = StrOutputParser()

chain = prompt1 | model | parser

result = chain.invoke({'topic': docs[0].page_content})

print(result)