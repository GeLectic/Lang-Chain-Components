from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
    return len(text.split())

chat_model = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen3-4B-Instruct-2507',
    task = 'text-generation',
    temperature=1.3
)

model = ChatHuggingFace(llm = chat_model)

prompt1 = PromptTemplate(
    template='Tell me a joke on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Rate this joke {topic} on a scale of 10',
    input_variables= ['topic']
)

parser = StrOutputParser()

joke_gen_chain = prompt1 | model | parser

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'count' : RunnableLambda(word_count)
})

final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke({'topic':'Software bug'})

print(result)
