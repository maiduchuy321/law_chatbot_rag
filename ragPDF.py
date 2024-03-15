from langchain_community.vectorstores import Chroma
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from rag import RAG
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        # self.model = RAG.load_model()
        self.model = LlamaCpp(model_path= "I:/Hoc/do an/model/ggml-vistral-7B-chat-q4_1.gguf",
                                temperature=0,
                                n_ctx=2048,
                                max_tokens=2000,
                                top_p=1,
                                n_gpu_layers=20,
                                n_batch=1024,
                                verbose=True,)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(template = """<s>[INST]  <<SYS>>\n Bạn là trợ lý cho các nhiệm vụ trả lời câu hỏi.
            Sử dụng các phần ngữ cảnh đã được truy xuất sau đây để trả lời câu hỏi. context: {context}
            Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết. Dùng ba câu tối đa và giữ câu trả lời ngắn gọn. <</SYS>> \n\n
            <s>[INST]  question: {question} 
            Answer: [/INST]""")
        # self.pdf_file_path = pdf_file_path

    # def vectorize(self,pdf_file_path: str):   
    #     docs = PyPDFLoader(file_path=pdf_file_path).load()
    #     chunks = self.text_splitter.split_documents(docs)
    #     chunks = filter_complex_metadata(chunks)
    #     # print(chunks)
    #     vector_store = Chroma.from_documents(documents=chunks, embedding=HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert"))
    #     return vector_store
    
    # def pdf_chain(self, prompt, db):
    #     self.retriever = db.as_retriever(search_type="similarity", 
    #                                      search_kwargs = {"k":3}, 
    #                                      max_tokens_limit=1024)
    #     self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
    #                   | prompt
    #                   | self.model
    #                   | StrOutputParser())
    
    # def generate_answer(self, query):
    #     template = """<s>[INST]  <<SYS>>\n Bạn là trợ lý cho các nhiệm vụ trả lời câu hỏi. Sử dụng các phần ngữ cảnh đã được truy xuất sau đây để trả lời câu hỏi. 
    #         Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết. Dùng ba câu tối đa và giữ câu trả lời ngắn gọn. <</SYS>> \n\n
    #         <s>[INST]  Question: {question} 
    #         Context: {context} 
    #         Answer: [/INST]"""
        
    #     prompt = PromptTemplate(template=template, input_variables=["query"])
    #     # vector_store = self.vectorize()
    #     db = self.vectorize(self.pdf_file_path)
    #     pdf_chain = self.pdf_chain(prompt, db)
    #     response = pdf_chain.invoke(query)
    #     return response
                
    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert"))
        self.retriever = vector_store.as_retriever(search_type="mmr",
                                                   search_kwargs={"k": 3},
                                                   )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None