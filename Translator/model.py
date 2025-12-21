import kagglehub
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer,Seq2SeqTrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch
import os


MODEL = "Helsinki-NLP/opus-mt-en-zh"
MAX_LENGTH = 128
os.environ["KAGGLEHUB_CACHE"] = "D:/kaggle_cache"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
  
def main(): 
    path = kagglehub.dataset_download("qianhuan/translation")
    file_path = os.path.join(path,"translation2019zh")
  
    train_file = os.path.join(file_path,"translation2019zh_train.json")
    valid_file = os.path.join(file_path,"translation2019zh_valid.json")
    
    try:
        raw_train_dataset = load_dataset("json", data_files=train_file, split="train", streaming=True)
        raw_test_dataset = load_dataset("json", data_files=valid_file, split="train", streaming=True)
        
        # create dataset
        tokenized_train_dataset = raw_train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["english","chinese"]
        )
        
        tokenized_test_dataset = raw_test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["english","chinese"]
        )
        training_args = Seq2SeqTrainingArguments(
        output_dir="./Translator/results",
        fp16=True, 
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./Translator/logs",
        logging_steps=100,
        load_best_model_at_end=True,
        greater_is_better=False,
        save_total_limit=2,
        warmup_steps=500,
        max_steps=5000,
        predict_with_generate=True
        )
        
        trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset.take(1000),
        )
        
        print("Starting training...")
        trainer.train()
        
        test_outputs = trainer.predict(tokenized_test_dataset.take(100))
        print("\nTest Output:\n")
        print(test_outputs)
        
        trainer.save_model("./Translator/final_model")
        tokenizer.save_pretrained("./Translator/final_model")
        
        test_sentences = [
            "The teacher is reading a book in the library.",
            "Although it was raining heavily, the football match continued.",
            "That exam was a piece of cake!"
        ]

        for s in test_sentences:
            print(f"EN: {s}")
            print(f"ZH: {translate(s)}")
            print("-" * 10)

    except Exception as e:
        print(f"Error: {e}")
        
def preprocess_function(examples):
    inputs = examples["english"]
    outputs = examples["chinese"]
    
    model_inputs = tokenizer(inputs,max_length=MAX_LENGTH,truncation=True,padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(outputs, max_length=MAX_LENGTH, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"] 
    return model_inputs

def translate(inputs):
    token = tokenizer(inputs, return_tensors="pt",padding=True).to(device)
    generated = model.generate(**token)
    output = tokenizer.decode(generated[0],skip_special_tokens=True)
    return output

if __name__ == "__main__":
    tokenizer = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL).to(device) 
    main()

    
    