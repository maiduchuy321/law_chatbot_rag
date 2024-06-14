from langchain_community.llms import LlamaCpp

class loadModel:
    _instance = None  # Biến lưu trữ thể hiện duy nhất của lớp

    def __new__(cls, model_name, n_gpu_layers, n_batch, temperature, max_tokens, top_p):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.n_gpu_layers = n_gpu_layers
            cls._instance.n_batch = n_batch
            cls._instance.temperature = temperature
            cls._instance.max_tokens = max_tokens
            cls._instance.top_p = top_p
            cls._instance.llm = None  # Khởi tạo biến llm là None
        elif cls._instance.model_name != model_name or \
             cls._instance.n_gpu_layers != n_gpu_layers or \
             cls._instance.n_batch != n_batch or \
             cls._instance.temperature != temperature or \
             cls._instance.max_tokens != max_tokens or \
             cls._instance.top_p != top_p:
            # Nếu có tham số mới, cập nhật lại các tham số
            cls._instance.model_name = model_name
            cls._instance.n_gpu_layers = n_gpu_layers
            cls._instance.n_batch = n_batch
            cls._instance.temperature = temperature
            cls._instance.max_tokens = max_tokens
            cls._instance.top_p = top_p
            cls._instance.llm = None  # Reset lại biến llm
        return cls._instance

    def load_model_llm(self):
        if self.llm is None:
            # Load model nếu chưa được load
            self.llm = LlamaCpp(
                model_path=self.model_name,
                temperature=self.temperature,
                n_ctx=2048,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                n_gpu_layers=self.n_gpu_layers,
                n_batch=self.n_batch,
                verbose=True,
            )
        return self.llm
