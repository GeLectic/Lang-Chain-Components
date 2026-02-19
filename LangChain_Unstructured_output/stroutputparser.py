from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

response_schemas = [
    ResponseSchema(name="summary", description="A 5 line summary of the topic"),
    ResponseSchema(name="fact", description="One interesting fact about the topic")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

template = PromptTemplate(
    template="Write information about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'Black hole'})

print(result)