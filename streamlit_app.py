import streamlit as st
from typing import Generator
from RAG import get_response


st.set_page_config(page_icon="💬", layout="wide",
                   page_title="AskMeBot GenAI App",
                   initial_sidebar_state="expanded")
                   #menu_items={"About": "Built by @suresh"})

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("🤖")

    # upload a PDF file
#st.file_uploader("Upload your PDF", type='pdf')

st.subheader("AskMe_Bot", divider="rainbow", anchor=False)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


if question := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(question)
        with st.spinner(text="Thinking..."):
            try:
                response = get_response(question)
            except Exception as e:
                st.stop()
    with st.chat_message("assistant", avatar='🤖'):
        full_response = ""
        placeholder = st.empty()
        
        for chunk in response.split():
            full_response += chunk + " "
        placeholder.markdown(response)