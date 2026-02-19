from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate

load_dotenv()

chat_model = HuggingFaceEndpoint(
    repo_id = 'HuggingFaceH4/zephyr-7b-beta',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = chat_model)

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_varibales =  ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a Linkedin Post about {topic}',
    input_varibales = ['topic']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': prompt1 | model | parser,
    'linkedin': prompt2 | model | parser
})

result = parallel_chain.invoke({'topic':'Graph RAG'})
print(result['tweet'])

print(result['linkedin'])