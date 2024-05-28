
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class AutoModel():
    def __init__(self, model_name,device ="cuda"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        print(torch.cuda.device_count())
        self.device = torch.device(device)
        print(device)
        
        self.model.to(self.device)        
        

    def answer(self,message: str) -> tuple:
        # Load tokenizer and model

        # Encode the input prompt
        inputs = self.tokenizer.encode(message, return_tensors='pt', truncation=True, return_attention_mask=True).to(self.device)

        # Generate response using the model with log probability calculation
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=50,output_logits=True, return_dict_in_generate=True)


        log_probs = []
        ids = []
        for logits in outputs.logits:
            # Calculate log probabilities
            logits.to("cpu")
            log_probs.append(torch.max(torch.nn.functional.log_softmax(logits, dim=1)).item())
            ids.append(torch.argmax(logits, dim=1).item())
        
        

        # Convert token ids to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        answer = self.tokenizer.decode(ids)

        # Collect the probabilities
        probs = []
        for token, p in zip(tokens, np.exp(log_probs)):
            probs.append((token, p))

        # Perplexity calculation
        perplexity = np.exp(-np.mean(log_probs))

        return answer, perplexity, probs

if __name__ == "__main__":
    #Test the model Llama 7B
    model = AutoModel("/dtu/blackhole/01/138401/Meta-Llama-3-8B",device="cuda")
    print("Model loaded")
    message = "What is the capital of France?"
    print(model.answer(message))