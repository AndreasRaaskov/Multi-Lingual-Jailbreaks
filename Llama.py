
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import os
import time
import datetime

class AutoModel():
    def __init__(self, model_name,device ="cuda"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        print(torch.cuda.device_count())
        self.device = torch.device(device)
        print(device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        torch.cuda.empty_cache()  # Free unused memory before loading model
        self.model.to(self.device) 
        

    def answer(self,message: list[str]) -> tuple:
        # Load tokenizer and model


        # Encode the input prompt
        inputs = self.tokenizer.batch_encode_plus(message, return_tensors='pt', padding=True, truncation=True, return_attention_mask=True).to(self.device)

        # Generate response using the model with log probability calculation
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=500,output_logits=True,pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True)


        # Process logits for each response in the batch
        answers = []
        perplexities = []
        probs_list = []
        print(outputs.keys())
        for i in range(outputs.sequences.shape[0]):
            print(outputs.logits)
            print(outputs.logits[i].shape)
            print(outputs.sequences)
            sequence_logits = outputs.logits[i]  # scores for each token generated in sequence i
            sequence_ids = outputs.sequences[i]  # token ids of sequence i

            # Calculate log probabilities and collect token-level outputs
            log_probs = []
            ids = sequence_ids.tolist()
            for logits in sequence_logits:
                logits = logits.to("cpu")
                log_prob = torch.max(torch.nn.functional.log_softmax(logits, dim=-1))
                log_probs.append(log_prob.item())

            # Convert token ids to tokens and decode
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            answer = self.tokenizer.decode(ids,skip_special_tokens=True)

            # Collect the probabilities for each token
            probs = [(token, np.exp(log_prob)) for token, log_prob in zip(tokens, log_probs)]

            # Calculate perplexity for the sequence
            perplexity = np.exp(-np.mean(log_probs))

            # Append results for this sequence
            answers.append(answer)
            perplexities.append(perplexity)
            probs_list.append(probs)
        """ 
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
        """   
        return answers, perplexities, probs_list
    
def Llama_answer_pipeline(LLM_name,language,translation_model):


    #Load the data
    #data = pd.read_csv(os.path.join("translations",language[:3]+"_"+translation_model+".csv")).loc[:,["id","question translation"]]

    data = pd.read_csv(os.path.join("Google_translations",language[:3]+"_"+translation_model+".csv")).loc[:,["id","question translation"]]
    
    # Define the model
    if LLM_name=="Llama-3-8B":
        model = AutoModel("/dtu/blackhole/01/138401/Meta-Llama-3-8B") #ToDo find more elegant way than hardcode path
    elif LLM_name=="Llama-3-70B":
        model = AutoModel("/dtu/blackhole/01/138401/Meta-Llama-3-70B") #ToDo find more elegant way than hardcode path

    
    

    #Loop through the questions and get the answers
    q_list=list(data["question translation"])
    answers = []
    perplexities = []
    probs = []
    
    batch_size = 5
    for i in range(0, len(q_list), batch_size):
        print(f"Current Time: {datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")
        answer, perplexity, prob = model.answer(q_list[i:i + batch_size])
        answers.extend(answer)
        perplexities.extend(perplexity)
        probs.extend(prob)
    
    data.loc[:,"answers"] = answers
    data.loc[:,"perplexity"] = perplexities
    data.loc[:,"probs"] = probs

    #Save the data
    data.to_csv(os.path.join("Results",language[:3]+"_"+LLM_name+"_"+translation_model+".csv"), index=False)

if __name__ == "__main__":
    language_list = ["arb_Arab","hat_Latn"]#["tha_Thai","zul_Latn","zho_Hans","glg_Latn"]# ,"arb_Arab","hat_Latn"
    #language_list =["dan_Latn"]
    translation_model = "Google"

    for language in language_list:
        
        
        Llama_answer_pipeline("Llama-3-8B",language,translation_model)