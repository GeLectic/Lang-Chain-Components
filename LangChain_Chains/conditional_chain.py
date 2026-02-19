from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task = "text-generation"
)

model1 = ChatHuggingFace(llm = llm)

model2 = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

class Feedback(BaseModel):

    sentiment : Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'Classify  the sentiment of the following feedback  into the positive or negative \n {feedback} \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables= {'format_instruction': parser2.get_format_instructions()}
)


parser = StrOutputParser()

classifier_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
    template= 'Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template= 'Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)



branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model1 | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model1 | parser ),
    RunnableLambda(lambda x: "could not find sentimnet or it is neutral")    
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback':"What a terrible application this is "}))
