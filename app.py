import streamlit as st
from langsmith import Client
import os
from rag import RAG
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory

#sidebar
with st.sidebar:
    st.title('🦙💬Chatbot')

    # Set LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # Add the toggle for LangSmith API key source
    use_secret_key = st.sidebar.toggle(label="Using LangSmith", value=False)

    # Conditionally set the project name based on the toggle
    if not use_secret_key:
        os.environ["LANGCHAIN_PROJECT"] = "Streamlit Demo"
    else:
        project_name = st.sidebar.text_input(
            "Name your LangSmith Project:", value="Streamlit Demo"
        )
        os.environ["LANGCHAIN_PROJECT"] = project_name

    # Conditionally get the API key based on the toggle
    if not use_secret_key:
        langchain_api_key = None  # assuming it's stored under this key in secrets
    else:
        langchain_api_key = st.sidebar.text_input(
            "👇 Add your LangSmith Key",
            value="",
            placeholder="Your_LangSmith_Key_Here",
            label_visibility="collapsed",
        )

        st.info("⚠️ Get your [LangSmith API key](https://python.langchain.com/docs/guides/langsmith/walkthrough).")
    
    langchain_endpoint = "https://api.smith.langchain.com"
    if langchain_api_key is not None:
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

    
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = "some_initial_value"


    #Select model and parameters
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

    #Clear chat history
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]
        memory.clear()
    st.button('Xóa lịch sử cuộc trò chuyện', on_click=clear_chat_history)

    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"



#main panel
st.title("💬 Chatbot hỏi đáp văn bản pháp luật")
st.markdown("___")
# Check if the LangSmith API key is provided
if not langchain_api_key or langchain_api_key.strip() == "Your_LangSmith_Key_Here":
    st.info("⚠️ Add your [LangSmith API key](https://python.langchain.com/docs/guides/langsmith/walkthrough) to continue, or switch to the Demo key")
else:
    client = Client(api_url=langchain_endpoint, api_key=langchain_api_key)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# memory = ConversationBufferMemory(
#     chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
#     return_messages=True,
#     memory_key="chat_history",
# )
        
rag = RAG(llm, n_gpu_layers_option, n_batch_option, temperature_option, max_length_option, top_p_option)
# llm_load = rag.load_model()
#User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag.generate_llm_response(prompt)
            placeholder = st.empty()
            full_response = ''
            # print(response)
            for key, value in response.items():
                if key == 'result':
                    full_response += value
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


