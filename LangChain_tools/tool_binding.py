from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.messages import HumanMessage
from dotenv import load_dotenv
import requests

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

# create tool 

@tool
def multiply(a: int, b: int) ->int:
    "Give 2 number a and b this tool return the product"
    return a*b

print("---- Tool is Created ----")

# binding the tool 

model_with_tool =  model.bind_tools([multiply])

print("---- Tool is binded Successfully ----")

print("enter the query")
query = str(input())

messages = [query]

result = model_with_tool.invoke(messages)

messages.append(result)

print("---- Tool is Executed Successfully ----")

tool_res = multiply.invoke(result.tool_calls[0])

messages.append(tool_res)

final_output = model_with_tool.invoke(messages)

print(final_output.content)