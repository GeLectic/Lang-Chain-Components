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
    template = 'Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Summarize the following text {text}',
    input_variables= ['text']
)

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = report_gen_chain | branch_chain 

result = final_chain.invoke({'topic': 'Global Warming'})

print(result)