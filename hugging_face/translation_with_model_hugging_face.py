# This file  loads a Hugging Face translation model for English→French
# and uses it to translate the input sentence
# Supported by transformer 4.x.x versions

from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()
import os


HF_TOKEN= os.getenv("HF_TOKEN")

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-fr"
)

text = "Artificial intelligence is transforming many industries."

result = translator(text)

print(result[0]["translation_text"])