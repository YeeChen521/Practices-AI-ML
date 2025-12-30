import json
import torch
import requests
import re
import numpy as np
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

load_dotenv()

MODEL_PATH = "./final_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completionsL"

JOB_LIST = ['HR', 'designer', 'Information-Technology',
       'Teacher', 'Advocate', 'Business-Development',
       'Healthcare', 'Fitness', 'Agriculture', 'BPO', 'Sales', 'Consultant',
       'Digital-Media', 'Automobile', 'Chef', 'Finance',
       'Apparel', 'Engineering', 'Accountant', 'Construction',
       'Public-Relations', 'Banking', 'Arts', 'Aviation']

print("Loading model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

def predict_category(resume_text):
    inputs = tokenizer(resume_text,max_length=512, truncation=True,padding="max_length", return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    probs = F.softmax(logits,dim=-1)
    pred_idx = torch.argmax(probs,dim=-1).item()
    confidence = probs[0,pred_idx].item()
    
    return JOB_LIST[pred_idx],confidence

def build_prompt(resume,jd,category,confidence):
    return f"""
    You are an AI resume evaluation assistant.
    The resume was classified into the category "{category}"
    with confidence {confidence: .2f}.
    This classification is approximate and should be used only as guidance.
    Analyze the resume again the job descripton.
    
    Resume:
    \"\"\"{resume}\"\"\"
    
    Job Description:
    \"\"\"{jd}\"\"\"
    
    Return ONLY valid JSON in this format:
    {{
        "match_score": number between 0 and 100,
        "strengths": [string],
        "missing_skills": [string],
        "resume_improvements": [string],
        "overall_feedback": [string]
    }}
    
    IMPORTANT:
    - Return ONLY a raw JSON object
    - Do NOT add explanations
    - Do NOT use markdown
    - Do NOT wrap in ```json
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
        "temperature": 0.3
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

print("\n=== Resume–JD Matching Assistant ===\n")

resume_text = input("Paste your resume here: \n\n")
jd_text = input("\nPaste the job description here: \n\n")

category,confidence = predict_category(resume_text)
print("\n=== Resume Classification ===")
print(f"Predicted Category: {category}")
print(f"Confidence: {confidence*100:.2f}%")
if confidence < 0.5:
    print("Prediction confidence is low (approximate category)")
    
prompt = build_prompt(resume_text, jd_text, category, confidence)
feedback = call_openrouter(prompt)
print("=== Job Match Feedback ===")
print(f"Match Score: {feedback['match_score']}%")

print("\nStrengths:")
for s in feedback["strengths"]:
    print("-", s)
    
print("\nMissing Skills:")
for m in feedback["missing_skills"]:
    print("-", m)
    
print("\nResume Improvement Suggestions:")
for r in feedback["resume_improvements"]:
    print("-", r)
    
print("\nOverall Feedback:")
print(feedback["overall_feedback"])
    