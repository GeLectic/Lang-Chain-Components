from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

url = "https://www.flipkart.com/apple-iphone-17-pro-cosmic-orange-256-gb/p/itm76fe37ca9ea8c"

chat_model = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen3-4B-Instruct-2507',
    task = 'text-generation',
    temperature=1.5
)

model = ChatHuggingFace(llm = chat_model)

prompt1 = PromptTemplate(
    template = ' what are the review of this smart phone, and  should one with budget contriant but this  \n {topic}',
    input_variables=['topic']
)

loader = WebBaseLoader(url)

docs = loader.load()

parser = StrOutputParser()

chain = prompt1 | model | parser

result = chain.invoke({'topic': docs[0].page_content})

print(result)

