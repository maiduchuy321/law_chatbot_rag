from openai import OpenAI
import streamlit as st

with st.sidebar:
    st.title('🦙💬Chatbot')

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['Vistral-7B-q4', 'Vistral-7B-q8', 'Vinallama-7b-q5'], key='selected_model')
    if selected_model == 'ggml-vistral-7B-chat-q4_1':
        llm = 'uonlp/Vistral-7B-Chat-gguf'
    elif selected_model == 'ggml-vistral-7B-chat-q8':
        llm = 'uonlp/Vistral-7B-Chat-gguf'
    else:
        llm = 'vilm/vinallama-7b-chat-GGUF'

    langsmith_api_key = st.text_input("LangSmith API Key", 
                                      key="langsmith_api_key", 
                                      type="password", 
                                      placeholder="Not required")
    "[Get an LangSmith API key](https://smith.langchain.com)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

st.title("💬 Chatbot hỏi đáp văn bản pháp luật")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)