from llama_cpp import Llama
from llama_cpp.llama_chat_format import format_llama2, format_chatml

class Llama2():

    def __init__(self):
        # Load the Llama2 7B model and tokenizer
        self.model_name = "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf"
        self.model_path = "/Users/reorganism/models/"
        
    def load(self) -> bool:
        
        self.model = Llama(
            model_path=f"{self.model_path}/{self.model_name}", 
            n_ctx = 4096,
            n_gpu_layers= -1
        )
        return True
    
    def run(self, prompt, **kwargs):
        #self.load()
        
        # Calls the model in the gpu
        # TODO if self.remote make this a websocket call
        output = self.model(
            prompt['prompt'],
            stop = prompt['stop'],
            temperature = kwargs.get('temperature', 0.9),
            top_p = kwargs.get('top_p', 0.95),
            repeat_penalty = kwargs.get('repeat_penalty', 1.2),
            top_k = kwargs.get('top_k', 50),
            max_tokens = kwargs.get('max_tokens', 4096*4),
            frequency_penalty = kwargs.get('frequency_penalty', 0.1),
            presence_penalty = kwargs.get('presence_penalty', 0.1),
            tfs_z = kwargs.get('tfs_z', 1.0),
            mirostat_mode=kwargs.get('mirostat_mode', 0),
            mirostat_tau=kwargs.get('mirostat_tau', 5.0),
            mirostat_eta=kwargs.get('mirostat_eta', 0.1),
        )
        # Fetches the preferred output
        text: str = output["choices"][0]["text"]
        text = text.replace(prompt['prompt'], "")
        return text