"""
Code for difining NLLB translator models. 


inspired by this notebook: https://colab.research.google.com/drive/1hpAjpSol7I4mMIFkZSJgMC4ZhfQZXaiU?usp=sharing#scrollTo=cRxj4eL0Wlh2
"""


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from datasets import load_dataset
import pandas as pd
import os 

def download_nllb(model_list):
    #DownLoad the models and put it in TranslationModels folder

    for model in model_list:
        model_name = "nllb"+model.split("-")[-1]
        
        #Check if the model is already downloaded
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(f"TranslationModels/{model_name}")
            tokenizer = AutoTokenizer.from_pretrained(f"TranslationModels/{model_name}")
        except:
            #Download the model
            tokenizer = AutoTokenizer.from_pretrained(model)
            model = AutoModelForSeq2SeqLM.from_pretrained(model)

            #Save the model
            model.save_pretrained(f"TranslationModels/{model_name}")
            tokenizer.save_pretrained(f"TranslationModels/{model_name}")



class nllb_translator:
    def __init__(self, model_name):
        #Check if GPU is available

        # Set environment variable to make only GPU 1 visible to this script
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #Load the model
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"TranslationModels/{model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(f"TranslationModels/{model_name}")

        #Move the model to the device
        print(f"Allocated memory: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")
        self.model.to(self.device)
        


    def translate(self,texts,source_language,target_language) -> list[str] | str | str: 

        #Tokenize the texts and add it to device. 
        print(f"Model alocated: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")
        inputs = self.tokenizer(texts, return_tensors="pt", padding = True).to(self.device)
        print("Tokens made")
        print(f"Tokens alocated memory: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")
        print(f"Cached memory: {torch.cuda.memory_reserved(self.device) / 1e9:.2f} GB")
        #Translate tokens on device and make a tokenized output that is sent to cpu
        translated_tokens = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_language]).to("cpu")

        #delete inputs
        del inputs

        #Transform output back to a list.
        output = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        return output




if __name__ == "__main__":
    # Load to models
    model_list = ["facebook/nllb-200-3.3B"]# ["facebook/nllb-200-distilled-600M", "facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-3.3B"]
    download_nllb(model_list)

    #from DoNotAnswer import get_do_not_answer_dataset
    #model = nllb_translator("nllb1.3B")
    #data = get_do_not_answer_dataset()
    #translation = model.translate(list(data["question"])[:10],None,target_language="dan_Latn")
    #print(translation)
