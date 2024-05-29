
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class AutoModel():
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def answer(self,message: str) -> tuple:
        # Load tokenizer and model

        # Encode the input prompt
        inputs = self.tokenizer.encode(message, return_tensors='pt')

        # Generate response using the model with log probability calculation
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=50,output_logits=True,pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True)


        log_probs = []
        ids = []
        for logits in outputs.logits:
            # Calculate log probabilities
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
    model = AutoModel("EleutherAI/gpt-neo-2.7B")
    print("Model loaded")
    message = "What is the capital of France?"
    print(model.answer(message))