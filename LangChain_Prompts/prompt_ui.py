from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

st.header('Research Tool')

paper_input = st.text_input('Enter the name of research paper to summarize')

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical Derivation"])

length_input = st.selectbox("Select Explanation Lenght", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (Very detailed Explanation)"])

template = load_prompt('template.json')


if st.button('Summarize'):

    chain = template | model
    result = chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    })
    st.write(result.content)