from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()

model  = HuggingFaceEndpoint(
      repo_id="deepseek-ai/DeepSeek-V3.2"
)

llm = ChatHuggingFace(llm = model)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools = [search_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
    handle_parsing_errors=True
)

response = agent_executor.invoke({"input":"Cheapest flight from the capaital of USA to the capital of India"})

print(response)
