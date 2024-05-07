# Repository for Predicting Multi-Lingual Jailbreaks

# Setup

## install venv


python3.10 -m venv env

.\env\Scripts\activate

pip install -r requirements.txt

note python 3.11 results in an error in the fairseq package that may be fixed in the future.

## Add keys.

Make a file called key.env

fill it up with

OPENAI_API_KEY = <your key>
GOOGLE_TRANSLATE_API_KEY = <your key>