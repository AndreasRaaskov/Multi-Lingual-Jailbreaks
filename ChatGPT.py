
import openai
import dotenv


MODEL_ID = "gpt-3.5-turbo-0301"
#MODEL_ID = "gpt-4-0613"

keys = dotenv.dotenv_values("keys.env")
openai.api_key = keys["OPENAI_API_KEY"]




def GPT_answer(message: str) -> str:
    #gives the response to the message and the logprobs of the tokens
    output=openai.chat.completions.create(
            model=MODEL_ID, messages=[{"role": "user", "content": message}],logprobs=True
        )
    tokens = []
    logprobs = []
    for item in output.choices[0].logprobs.content:
        tokens.append(item.token)
        logprobs.append(item.logprob)
    return tokens,logprobs

if __name__ == "__main__":
    print(GPT_answer("What is the meaning of life"))