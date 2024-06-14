import os
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

class RAG:
    def __init__(self,model_name, n_gpu_layers, 
                   n_batch, temperature, 
                   max_tokens, top_p):
        
        self.model_name = model_name  
        self.n_gpu_layers = n_gpu_layers  
        self.n_batch = n_batch  
        self.temperature = temperature  
        self.max_tokens = max_tokens  
        self.top_p = top_p

    #Load Model
    def load_model(self):
        #Load model
        llm =LlamaCpp (
            model_path= self.model_name,
            temperature= self.temperature,
            n_ctx=2048,
            max_tokens= self.max_tokens,
            top_p= self.top_p,
            n_gpu_layers= self.n_gpu_layers,
            n_batch= self.n_batch,
            verbose=True,
        )
        return llm

    # Read tu VectorDB
    def read_vectors_db(self):
        #Load Model Embedding
        embeddings=HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
        vector_db_path = "vectorstores/db_Chroma_json"
        db_chroma = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
        
        return db_chroma

    # Tao qa chain
    def create_qa_chain(self,prompt, llm, db):
        llm_chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type= "stuff",
            retriever = db.as_retriever(search_type="similarity", search_kwargs = {"k":3}),
            return_source_documents = True,
            chain_type_kwargs= {'prompt': prompt}
        )
        return llm_chain

    # Function for generating llm response
    def generate_llm_response(self, prompt_input):
        template = """<s>[INST]  <<SYS>>\n Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực.\n
        Hãy luôn trả lời một cách hữu ích nhất có thể.\n
        Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác.\n 
        Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.\n {context}<</SYS>> \n\n
        <s>[INST] {question} [/INST] \n """

        # Tao prompt template
        prompt = PromptTemplate(template=template, input_variables=["question"])

        #Tao chain
        db = self.read_vectors_db()
        llm_chain  =self.create_qa_chain(prompt, llm=self.load_model(), db= db)

        response = llm_chain.invoke({"query":prompt_input})
        return response




