
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np



def GPT_answer(message: str, model_name=r"/dtu/blackhole/01/138401/Meta-Llama-3-8B") -> tuple:
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Model loaded")
    model.eval()

    # Encode the input prompt
    inputs = tokenizer.encode(message, return_tensors='pt')

    # Generate response using the model with log probability calculation
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        log_probs = outputs.logits.log_softmax(dim=-1)
        predictions = torch.argmax(log_probs, dim=-1)

    # Convert token ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(predictions[0].tolist())

    # Extract log probabilities for the predicted tokens
    log_probs = log_probs.gather(2, predictions.unsqueeze(-1)).squeeze(-1)[0].tolist()

    # Perplexity calculation
    perplexity = np.exp(-np.mean(log_probs))

    return tokens, perplexity, log_probs

if __name__ == "__main__":
    print(GPT_answer("What is the meaning of life"))