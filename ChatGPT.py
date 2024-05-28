
import openai
import numpy as np

#MODEL_ID = "gpt-3.5-turbo-0301"
#MODEL_ID = "gpt-4-0613"




class GPT:
    def __init__(self, MODEL_ID):
        import dotenv
        keys = dotenv.dotenv_values("keys.env")
        openai.api_key = keys["OPENAI_API_KEY"]
        self.MODEL_ID = MODEL_ID
        

    def answer(self,message: str) -> str:
        #gives the response to the message and the perplexity of the sentence
        #Inspired by https://cookbook.openai.com/examples/using_logprobs 
        output=openai.chat.completions.create(
                model=self.MODEL_ID, messages=[{"role": "user", "content": message}],
                logprobs=True,
                temperature=0.0,
                max_tokens=200
            )
        tokens = []
        logprobs = []
        for item in output.choices[0].logprobs.content:
            tokens.append(item.token)
            logprobs.append(item.logprob)
        
        answer = output.choices[0].message.content
        
        #perplexity calculation
        perplexity = np.exp(-np.mean(logprobs))

        #probability calculation
        probs = []
        for token,p in zip(tokens,np.exp(logprobs)):
            probs.append((token,p))

        return answer,perplexity,probs
