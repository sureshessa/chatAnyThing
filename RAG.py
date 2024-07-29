from helper import *
from langchain_groq import ChatGroq
import streamlit as st
from streamlit_chat import message
import os

api_key = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192"
)

documents = get_PDF_data()
text_chunks = get_textchunks(documents)
vector_db = get_vector_store(text_chunks)

retriever = vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.35, "k": 5}
            )

qa_chain=get_chain(llm, retriever)

def get_response(question):
    return qa_chain.run({"query": question})

#print(qa_chain.run({"query": "Explain about Modeling Rainwater Harvesting Systems with Covered Storage Tank on A Smartphone"}))



    