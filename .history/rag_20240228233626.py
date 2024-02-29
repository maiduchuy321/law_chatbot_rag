from langchain_community.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os


class RAG:
    #Load Model
    def load_model(model_name, n_gpu_layers, 
                   n_batch, temperature, 
                   max_tokens, top_p):
        # Cau hinh
        model_file = model_name
        #Load model
        llm =LlamaCpp (
            model_path=model_file,
            temperature=temperature,
            n_ctx=2048,
            max_tokens=max_tokens,
            top_p=top_p,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=True,
        )
        return llm
    
    #Load Model Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")

    # Read tu VectorDB
    vector_db_path = "vectorstores/db_Chroma_json"
    db_chroma = Chroma(persist_directory=vector_db_path, embedding_function=embedding_model)
    
    #Retrieve
    retriever = db_chroma.as_retriever()

    



