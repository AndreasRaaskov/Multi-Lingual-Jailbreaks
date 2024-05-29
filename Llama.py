
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import os

class AutoModel():
    def __init__(self, model_name,device ="cuda"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        print(torch.cuda.device_count())
        self.device = torch.device(device)
        print(device)

        torch.cuda.empty_cache()  # Free unused memory before loading model
        self.model.to(self.device) 
        

    def answer(self,message: str) -> tuple:
        # Load tokenizer and model

        # Encode the input prompt
        inputs = self.tokenizer.encode(message, return_tensors='pt', truncation=True, return_attention_mask=True).to(self.device)

        # Generate response using the model with log probability calculation
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=500,output_logits=True, return_dict_in_generate=True)


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
    
def Llama_answer_pipeline(LLM_name,language,translation_model):


    #Load the data
    data = pd.read_csv(os.path.join("translations",language[:3]+"_"+translation_model+".csv")).loc[:,["id","question translation"]]
    
    # Define the model
    if LLM_name=="Llama-3-8B":
        model = AutoModel("/dtu/blackhole/01/138401/Meta-Llama-3-8B") #ToDo find more elegant way than hardcode path
    

    #Loop through the questions and get the answers
    q_list=list(data["question translation"])
    answers = []
    perplexities = []
    probs = []
    
    for q in q_list:
        answer, perplexity, prob = model.answer(q)
        answers.append(answer)
        perplexities.append(perplexity)
        probs.append(prob)
    
    data.loc[:,"answers"] = answers
    data.loc[:,"perplexity"] = perplexities
    data.loc[:,"probs"] = probs

    #Save the data
    data.to_csv(os.path.join("Results",language[:3]+"_"+LLM_name+"_"+translation_model+".csv"), index=False)

if __name__ == "__main__":
    language_list = ["dan_Latn","ban_Latn","hin_Deva","vie_Latn","por_Latn","tha_Thai","zul_Latn","zho_Hans","glg_Latn","arb_Arab"]
    #language_list =["dan_Latn"]
    translation_model = "Google"
    for language in language_list:
        Llama_answer_pipeline("Llama-3-8B",language,translation_model)