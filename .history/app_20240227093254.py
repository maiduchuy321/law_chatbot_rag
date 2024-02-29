from openai import OpenAI
import streamlit as st
import os

#sidebar
with st.sidebar:
    st.title('🦙💬Chatbot')

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['Vistral-7B-q4', 'Vistral-7B-q8', 'Vinallama-7b-q5'], key='selected_model')
    if selected_model == 'ggml-vistral-7B-chat-q4_1':
        llm = 'uonlp/Vistral-7B-Chat-gguf/ggml-vistral-7B-chat-q4_1.gguf'
    elif selected_model == 'ggml-vistral-7B-chat-q8':
        llm = 'uonlp/Vistral-7B-Chat-gguf/ggml-vistral-7B-chat-q8.gguf'
    else:
        llm = 'vilm/vinallama-7b-chat-GGUF'

    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=2048, value=512, step=8)

    langsmith_api_key = st.text_input("LangSmith API Key", 
                                      key="langsmith_api_key", 
                                      type="password", 
                                      placeholder="Not required")
    "[Get an LangSmith API key](https://smith.langchain.com)"

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]
    st.button('Xóa lịch sử cuộc trò chuyện', on_click=clear_chat_history)

    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

#Body
st.title("💬 Chatbot hỏi đáp văn bản pháp luật")
# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

os.environ['langsmith_api_key'] = langsmith_api_key

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])



# if prompt := st.chat_input():
#     if not openai_api_key:
#         st.info("Please add your OpenAI API key to continue.")
#         st.stop()

#     client = OpenAI(api_key=openai_api_key)
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)
#     response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
#     msg = response.choices[0].message.content
#     st.session_state.messages.append({"role": "assistant", "content": msg})
#     st.chat_message("assistant").write(msg)