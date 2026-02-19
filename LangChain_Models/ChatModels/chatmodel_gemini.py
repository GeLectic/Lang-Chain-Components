from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chatmodel = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 1.5)

results = chatmodel.invoke("write 5 lines poem on cricket")

print(results.content)