from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate

load_dotenv()

chat_model = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen3-4B-Instruct-2507',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = chat_model)

prompt1 = PromptTemplate(
    template='Tell me a joke on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Rate this joke {topic} on a scale of 10',
    input_varibales= ['topic']
)

parser = StrOutputParser()

joke_gen_chain = prompt1 | model | parser

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'rate': prompt2 | model | parser
})

final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke("Programming in C++")

print("Joke is \n ", result['joke'])
print("Rating of joke  is \n ", result['rate'])