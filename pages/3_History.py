# Import necessary libraries
import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import format_document

# Define the path for generated embeddings
DB_Chroma_PATH ='vectorstores/db_Chroma_json'
# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "chatPDF"
os.environ["LANGCHAIN_API_KEY"] = "API_KEY"
# Load the model of choice
def load_llm():
    llm = LlamaCpp(model_path= "I:/Hoc/do an/model/ggml-vistral-7B-chat-q4_1.gguf",
                                temperature=0,
                                n_ctx=2048,
                                max_tokens=2000,
                                top_p=1,
                                n_gpu_layers=20,
                                n_batch=1024,
                                verbose=True)
    return llm

# Set the title for the Streamlit app
st.title("Vistral Chat PDF - 🦜🦙")

# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type=["pdf"])

# Handle file upload
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    # Load CSV data using CSVLoader
    docs = PyPDFLoader(file_path=tmp_file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    chunks = filter_complex_metadata(chunks)


    # Create embeddings using Sentence Transformers
    embeddings=HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")

    # Create a FAISS vector store and save embeddings
    db = Chroma.from_documents(chunks, embeddings)
    # db.save_local(DB_Chroma_PATH)
    retriever=db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    # Load the language model
    llm = load_llm()

    _template = """<s>[INST]  <<SYS>>\nHãy tiếp tục cuộc trò chuyện với câu hỏi tiếp theo 
    và diễn đạt lại câu hỏi tiếp theo thành một câu hỏi độc lập, bằng ngôn ngữ gốc của nó.
    Chat History:
    {chat_history}
    Câu hỏi: {question}
    Câu hỏi độc lập: <</SYS>> \n\n"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    
    #Tạo PROMT trả lời câu hỏi mới nhất
    template = """<s>[INST]  <<SYS>>\nBạn là trợ lý cho các nhiệm vụ trả lời câu hỏi. \n
    Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi: {context}\n
    Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời. \n
    <</SYS>> \n\n<s>[INST] {question}[/INST] """

    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)
    
    memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )
    #Tao các chain và kết hợp chúng
    # First we add a step to load memory
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )
    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm,
        "docs": itemgetter("docs"),
    }
    # And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
    # Function for conversational chat
    def conversational_chat(query):
        result = final_chain.invoke({"question": query, "chat_history": st.session_state['history']})
        # print(result)
        st.session_state['history'].append((query, result))
        return result

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to csv data 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            # print(user_input)
            output = conversational_chat(user_input)
            print("Đây là output: \n", output["answer"])
            print("Đây là output doc: \n", output["docs"])
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output["answer"])

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")