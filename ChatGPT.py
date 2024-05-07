
import openai
import dotenv


#MODEL_ID = "gpt-3.5-turbo-0301"
MODEL_ID = "gpt-4-0613"

keys = dotenv.dotenv_values("keys.env")
openai.api_key = keys["OPENAI_API_KEY"]




def GPT_answer(message: str) -> str:
    return (
        openai.ChatCompletion.create(
            model=MODEL_ID, messages=[{"role": "user", "content": message}]
        )
        .choices[0]
        .message.content
    )
