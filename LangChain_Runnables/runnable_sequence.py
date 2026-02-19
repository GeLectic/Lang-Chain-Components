from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

prompt1 = PromptTemplate(
    template= 'Tell me a joke about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template= "Give the joke {joke} on a scale of 10 how would you rate this joke",
    input_variables=['joke']
)

parser = StrOutputParser()

#chain = prompt1 | model | parser | prompt2 | model | parser
chain = RunnableSequence(prompt1, model, parser, prompt2,model, parser)
result  = chain.invoke({'topic':'Programming'})

print(result)
