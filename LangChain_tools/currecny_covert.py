from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.messages import HumanMessage
import requests
import json
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetches the currency conversion factor between a base currency and a target currency."""
    url = f"https://v6.exchangerate-api.com/v6/f96090bbe5f329833d91e4fc/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: float, conversion_rate: float) -> float:
    """
    Calculates the target currency value.
    Requires the amount and the conversion rate found previously.
    """
    return base_currency_value * conversion_rate

# Bind tools (Removed InjectedToolArg so model sees the parameter)
model_with_tools = model.bind_tools([get_conversion_factor, convert])

query = HumanMessage("What is the conversion factor between USD and INR and based on that can you convert 10 usd to inr")
messages = [query]

# --- Step 1: Model decides to call the first tool ---
ai_msg_1 = model_with_tools.invoke(messages)
messages.append(ai_msg_1)

# Execute the tools requested in Step 1
for tool_call in ai_msg_1.tool_calls:
    if tool_call['name'] == 'get_conversion_factor':
        print(f"--- Calling Tool: {tool_call['name']} ---")
        tool_msg = get_conversion_factor.invoke(tool_call)
        messages.append(tool_msg)
        
# --- Step 2: Model sees the rate and decides to call the second tool ---
# Now that the history contains the conversion rate, the model knows it can proceed.
ai_msg_2 = model_with_tools.invoke(messages)
messages.append(ai_msg_2)

# Execute the tools requested in Step 2
for tool_call in ai_msg_2.tool_calls:
    if tool_call['name'] == 'convert':
        print(f"--- Calling Tool: {tool_call['name']} ---")
        tool_msg = convert.invoke(tool_call)
        messages.append(tool_msg)

# --- Step 3: Final Answer ---
final_response = model_with_tools.invoke(messages)
print("\nFinal Answer:")
print(final_response.content)