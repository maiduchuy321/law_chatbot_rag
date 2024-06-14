from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex
import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
# from langsmith import traceable

class RAG:
    def __init__(self, model_name, n_gpu_layers, 
                   n_batch, temperature, 
                   max_tokens, top_p) -> None:
      
      self.model_name = model_name  
      self.n_gpu_layers = n_gpu_layers
      self.n_batch = n_batch  
      self.temperature = temperature  
      self.max_tokens = max_tokens  
      self.top_p = top_p                
      
      self.embeddings = self.load_embeddings()
      Settings.embed_model = self.embeddings
      
      self.client_weaviate = weaviate.connect_to_local()  
      self.retriever = self.load_retriever()
      self.retriever_QA = self.load_retriever_QA()
      self.pipe = self.load_model_pipeline()
      self.prompt = self.load_prompt_template()
      self.rag_pipeline = self.load_rag_pipeline(llm=self.pipe,
                                            retriever=self.retriever,
                                            prompt=self.prompt)
    # @traceable
    def load_embeddings(self):
      embeddings = HuggingFaceEmbedding(
          model_name="maiduchuy321/vietnamese-bi-encoder-fine-tuning-for-law-chatbot",
      )
      return embeddings

    # @traceable
    def load_retriever(self):   
        retriever=None
        vector_store = WeaviateVectorStore(
            weaviate_client=self.client_weaviate,
            index_name="HCC_with_Vietnamese_Bi_Encoder_Fine_Tuning_For_Law_Chatbot")
        retriever = VectorStoreIndex.from_vector_store(vector_store)
        return retriever

    def load_retriever_QA(self):   
        retriever_QA=None
        vector_store_QA = WeaviateVectorStore(
            weaviate_client=self.client_weaviate,
            index_name="HCC_QuestionAndAnswer")
        index_QA = VectorStoreIndex.from_vector_store(vector_store_QA)
        retriever_QA = index_QA.as_retriever(
            vector_store_query_mode="hybrid",
            similarity_top_k=3,
            alpha=0.75 #very similar to vector search
        )
        return retriever_QA
        # vector_nodes = retriever_QA.retrieve(query)
        
    # @traceable
    def load_model_pipeline(self):
        llm = LlamaCPP(
            # You can pass in the URL to a GGML model to download it automatically
            # model_url=model_url,
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            model_path="I:/Hoc/do an/model/ggml-vistral-7B-chat-q4_1.gguf",
            temperature=0.1,
            max_new_tokens=256,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=2048,
            # kwargs to pass to __call__()
            generate_kwargs={},
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            model_kwargs={"n_gpu_layers": 30},
            # transform inputs into Llama2 format
            # messages_to_prompt=messages_to_prompt,
            # completion_to_prompt=completion_to_prompt,
            verbose=True)
        return llm
        
    # @traceable
    def load_prompt_template(self):
      text_qa_template_str = """<s>[INST]  <<SYS>>\nBạn là trợ lý cho các nhiệm vụ trả lời câu hỏi cho dịch vụ hành chính công. \n
        Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi: {context_str}\n
        Câu trả lời được sinh ra phải ngắn gọn và đầy đủ ý\n
        Không tạo ra câu trả lời nằm ngoài phạm vi được hỏi\n
        Nếu bạn không biết câu trả lời hãy nói tôi không có thông tin cho câu hỏi của bạn, đừng cố tạo ra câu trả lời.\n
        <</SYS>> \n\n<s>[INST] {query_str}[/INST] """
        
      prompt = PromptTemplate(text_qa_template_str)
      return prompt
        
    # @traceable
    def load_rag_pipeline(self, llm, retriever, prompt):
      rag_pipeline = retriever.as_query_engine(
          text_qa_template=prompt,
          llm=llm,
          vector_store_query_mode="hybrid",
          similarity_top_k=3,
          alpha=0.75 #very similar to vector search
      )
      return rag_pipeline
        
    # @traceable(run_type="llm")
    def rag(self, query):
        vector_nodes = self.retriever_QA.retrieve(query)
        context = []
        if vector_nodes[0].score <= 0.1:
            context.append({'answer': vector_nodes[0].metadata['answer']})
            context.append({'metadata': vector_nodes[0].metadata})
        else:
            query_engine = self.rag_pipeline
            response = query_engine.query(query)
            context.append({'answer': response.response})
            context.append({'metadata': response.metadata})
        return context
