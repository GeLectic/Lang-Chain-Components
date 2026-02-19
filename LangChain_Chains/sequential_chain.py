from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= 'Generate a detailed report on \n {topic}',
    input_varibles = ['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 point summary on \n {text}',
    input_variables = ['text']
)


chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'cancer'})

print(result)