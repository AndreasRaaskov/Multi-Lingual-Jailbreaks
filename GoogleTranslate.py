from typing import List
import requests




class GoogleTranslate:
    def __init__(self):
        import dotenv
        keys = dotenv.dotenv_values("keys.env")
        self.translate_api_key = keys["GOOGLE_TRANSLATE_API_KEY"]

    def translate(self,text, source_language, target_language) -> List[str] | str  | str:
        translations = requests.post(
            "https://translation.googleapis.com/language/translate/v2",
            params={
                "q": text,
                "target": target_language,
                "format": "text",
                "source": source_language,
                "key": self.translate_api_key,
            },
        ).json()["data"]["translations"]
        return [translation["translatedText"] for translation in translations]