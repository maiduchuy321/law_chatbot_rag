from langchain_community.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
import os


class RAG:

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


