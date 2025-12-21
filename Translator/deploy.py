import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_PATH = "./Translator/final_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
model = MarianMTModel.from_pretrained(MODEL_PATH).to(device)

def translate(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True).to(device)
    output_tokens = model.generate(**tokens)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print("Model Loaded! Type 'quit' to exit.")
while True:
    user_input = input("English: ")
    if user_input.lower() == 'quit': break
    print(f"Chinese: {translate(user_input)}\n")
