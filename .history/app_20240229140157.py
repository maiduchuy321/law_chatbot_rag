from langchain_community.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
import streamlit as st
import os
from rag import load_model, generate_llm_response

#sidebar
with st.sidebar:
    st.title('🦙💬Chatbot')

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['ggml-vistral-7B-chat-q4_1', 'ggml-vistral-7B-chat-q8', 'Vinallama-7b-q5'], key='selected_model')
        # llm = 'uonlp/Vistral-7B-Chat-gguf/ggml-vistral-7B-chat-q4_1.gguf'
    if selected_model == 'ggml-vistral-7B-chat-q4_1':
        llm = 'I:/Hoc/do an/model/ggml-vistral-7B-chat-q4_1.gguf'
    elif selected_model == 'ggml-vistral-7B-chat-q8':
        llm = 'uonlp/Vistral-7B-Chat-gguf/ggml-vistral-7B-chat-q8.gguf'
    else:
        llm = 'vilm/vinallama-7b-chat-GGUF'

    temperature_option = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    n_gpu_layers_option = st.sidebar.slider('n_gpu_layers', min_value=1, max_value=40, value=40, step=1)
    n_batch_option = st.sidebar.slider('n_batch', min_value=64, max_value=2048, value=1024, step=8)
    top_p_option = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length_option = st.sidebar.slider('max_length', min_value=64, max_value=2048, value=2000, step=8)

    langsmith_api_key = st.text_input("LangSmith API Key", 
                                      key="langsmith_api_key", 
                                      type="password", 
                                      placeholder="Not required")
    "[Get an LangSmith API key](https://smith.langchain.com)"

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]
    st.button('Xóa lịch sử cuộc trò chuyện', on_click=clear_chat_history)

    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

#main panel
st.title("💬 Chatbot hỏi đáp văn bản pháp luật")

os.environ['langsmith_api_key'] = langsmith_api_key

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llm_response(prompt)
            placeholder = st.empty()
            full_response = ''
            print(response)
            for key, value in response.items():
                if key == 'text':
                    full_response += value
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
