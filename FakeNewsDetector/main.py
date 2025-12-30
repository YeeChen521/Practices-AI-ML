import json
import torch
import requests
import re
import numpy as np
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, BertForSequenceClassification
import torch.nn.functional as F

load_dotenv()

MODEL_PATH = "./FakeNewsDetector/final_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

LABEL_MAP = {
    0: "Fake",
    1: "Real"
}

print("Loading model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

def predict_category(news_text):
    inputs = tokenizer(news_text,max_length=512, truncation=True,padding="max_length", return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    probs = F.softmax(logits,dim=-1)
    pred_idx = torch.argmax(probs,dim=-1).item()
    confidence = probs[0,pred_idx].item()
    
    return LABEL_MAP[pred_idx],confidence

def build_prompt(news,category,confidence):
    return f"""
    You are an AI fake news detector assistant.
    The news was classified into the category "{category}"
    with confidence {confidence: .2f}.
    This classification is approximate and should be used only as guidance.
    Your task:
    - Analyze the news content
    - Judge whether the content aligns with the model's prediction
    
    News:
    \"\"\"{news}\"\"\"

    Return ONLY valid JSON in this format:
    {{
        "match_score": 0-100,
        "is_fake": true or false,
        "reason": "concise explanation"
    }}
    
    Definitions:
    - match_score: how strongly the content supports the model’s classification
    - is_fake: your final judgment after analysis
    - reason: brief justification (1–2 sentences max)

    IMPORTANT:
    - Output ONLY raw JSON
    - No markdown
    - No explanations outside JSON
    """

def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group()) 

def call_openrouter(prompt):
    header = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    response = requests.post(
        OPENROUTER_URL,
        headers=header,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"OpenRouter error: {response.text}")
    
    content = response.json()["choices"][0]["message"]["content"]
    
    return extract_json(content)

print("\n=== Fake News Detector ===\n")

news_text = input("Paste the news text here: \n\n")

while news_text.lower() != "quit":
    category,confidence = predict_category(news_text)
    print("\n=== News Classification ===")
    print(f"Predicted Category: {category}")
    print(f"Confidence: {confidence*100:.2f}%")
    if confidence < 0.5:
        print("Prediction confidence is low (approximate category)")
        
    prompt = build_prompt(news_text, category, confidence)
    feedback = call_openrouter(prompt)
    print("=== LLM Feedback ===")
    print(f"Match Score: {feedback['match_score']}%")
        
    print("\n Is Fake :")
    print(feedback["is_fake"])

    print("\n Reason :")
    print(feedback["reason"])
    
    news_text = input("Paste the news text here: \n\n")
    