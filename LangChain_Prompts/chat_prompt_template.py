from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a Helpful {domain} expert'),
    ('human', 'Explain in simple Terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'chess', 'topic':'castle'})

print(prompt)