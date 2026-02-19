from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

class Review(TypedDict):
    summary: Annotated[str, 'A brief summary of the review']
    sentiment: Annotated[str, "Return sentiment of the review either negative, positive or neutral"]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated,
There are too many pre-installed applications that I cant remove. 
ALso the UI looks ugly and outdate in comparison to other brands. Hoping for new updates.
""")

print(result['summary'])
print(result['sentiment'])