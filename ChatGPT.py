
import openai
import dotenv
import numpy as np
import concurrent.futures
#MODEL_ID = "gpt-3.5-turbo-0301"
#MODEL_ID = "gpt-4-0613"

keys = dotenv.dotenv_values("keys.env")
openai.api_key = keys["OPENAI_API_KEY"]





class GPT:
    def __init__(self, MODEL_ID):
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
    
    def answer_batch(self,messages: str) -> str:
        #gives the response to the message and the perplexity of the sentence
        #Inspired by https://cookbook.openai.com/examples/using_logprobs 

        prompts = [{"role": "user", "content": message} for message in messages]
    
        # Define a function to perform the completion for one message
        def get_completion(prompt):
            return openai.chat.completions.create(
                model=self.MODEL_ID, 
                messages=[prompt],
                logprobs=True,
                temperature=0.0,
                max_tokens=200)
        
        # Use ThreadPoolExecutor to handle requests in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks to the executor
            futures = [executor.submit(get_completion, prompt) for prompt in prompts]

            # Collect results as they complete
            outputs = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        answers = []
        perplexitys = []
        probs_list = []
        for output in outputs:
            tokens = []
            logprobs = []
            for item in output.choice[0].logprobs.content:
                tokens.append(item.token)
                logprobs.append(item.logprob)
            
            
            #perplexity calculation
            perplexitys.append(np.exp(-np.mean(logprobs)))

            #probability calculation
            probs = []
            for token,p in zip(tokens,np.exp(logprobs)):
                probs.append((token,p))
            probs_list.append(probs)

            answers.append(output.choice[0].message.content)

        return answers,perplexitys,probs_list
