from langchain_community.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
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

    temperature_option = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p_option = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length_option = st.sidebar.slider('max_length', min_value=64, max_value=2048, value=512, step=8)

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

# Cau hinh
model_file = "I:/Hoc/do an/model/ggml-vistral-7B-chat-q4_1.gguf"
# Function for generating llm response
def generate_llm_response(prompt_input):
    string_dialogue = "<s>[INST]  <<SYS>>\n Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\nCâu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.<</SYS>> \n\n"
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    #Load model
    n_gpu_layers = 40
    n_batch = 1024
    llm =LlamaCpp(
        model_path=model_file,
        temperature=temperature_option,
        n_ctx=2048,
        max_tokens=max_length_option,
        top_p=top_p_option,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        verbose=True,
    )
    #Tao chain
    output = LLMChain(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


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