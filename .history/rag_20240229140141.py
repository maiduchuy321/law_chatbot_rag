from langchain_community.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os


#Load Model
def load_model(model_name, n_gpu_layers, 
                n_batch, temperature, 
                max_tokens, top_p):
    #Load model
    llm =LlamaCpp (
        model_path=model_name,
        temperature=temperature,
        n_ctx=2048,
        max_tokens=max_tokens,
        top_p=top_p,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        verbose=True,
    )
    return llm

# Function for generating llm response
def generate_llm_response(prompt_input):
    # Cau hinh
    llm_load = load_model(llm, n_gpu_layers_option, n_batch_option, temperature_option, max_length_option, top_p_option)

    template = """<s>[INST]  <<SYS>>\n Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực.\n
    Hãy luôn trả lời một cách hữu ích nhất có thể.\n
    Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác\n. 
    Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.<</SYS>> \n\n
    <s>[INST] {question} [/INST] \n """

    # Tao prompt template
    prompt = PromptTemplate(template=template, input_variables=["question"])

    #Tao chain
    llm_chain = LLMChain(prompt=prompt, llm=llm_load)

    response = llm_chain.invoke({"question":prompt_input})
    return response

# class RAG:
#     def __init__(self, model_name, n_gpu_layers, 
#                    n_batch, temperature, 
#                    max_tokens, top_p):
        
#         self.model_name= model_name
#         self.n_gpu_layers= n_gpu_layers
#         self.n_batch= n_batch
#         self.temperature= temperature
#         self.max_tokens= max_tokens
#         self.top_p= top_p

    
    
#     #Load Model Embedding
#     embedding_model = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")

#     # Read tu VectorDB
#     vector_db_path = "vectorstores/db_Chroma_json"
#     db_chroma = Chroma(persist_directory=vector_db_path, embedding_function=embedding_model)
    
#     #Retrieve
#     retriever = db_chroma.as_retriever()





