from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-2")


res = model.invoke("What is the meaning of life?")

print(res.content)