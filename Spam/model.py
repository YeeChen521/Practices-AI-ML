import kagglehub
from datasets import load_dataset, interleave_datasets
import os
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate

# Test Accuracy: 99.00%

"""
ALBERT uses a specific type of tokenization called SentencePiece.
It breaks words into sub-units("playing" becomes "play" + "ing") so 
it can understand words it hasn't seem before
"""

os.environ["KAGGLEHUB_CACHE"] = "D:/kaggle_cache"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
MODEL_NAME = "albert-base-v2"

# pre-loads the logic for calculating accuracy
metric = evaluate.load("accuracy")

def preprocess_function(examples):
    # every sentence is forced to be 128 tokens longer
    # shorter sentences are "padded" with zeros
    # truncation: ensures the model dont crash if it sees a very long comment
    model_inputs = tokenizer(examples["CONTENT"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = examples["CLASS"]
    return model_inputs

def compute_metrics(eval_pred):
    # logits: raw scores from the model
    logits, labels = eval_pred
    
    # picks the index of the highest score
    # converts the model's "confidence" into a "prediction"
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    path = kagglehub.dataset_download("lakshmi25npathi/images")
    file_list = ["Psy.csv", "KatyPerry.csv", "LMFAO.csv", "Eminem.csv", "Shakira.csv"]
    
    raw_streams = []
    for i, name in enumerate(file_list, 1):
        file_path = os.path.join(path, f"Youtube0{i}-{name}")
        # streaming: able to train on datasets that are larger than the computer memory
        # split: "take this file stream and treat it as my primary training source"
        data = load_dataset("csv", data_files=file_path, split="train", streaming=True)
        raw_streams.append(data)
    
    # mix the data in different file together
    full_dataset = interleave_datasets(raw_streams)
    
    shuffled_data = full_dataset.shuffle(seed=42, buffer_size=1000)
    
    test_size = 400
    test_dataset = shuffled_data.take(test_size)
    train_dataset = shuffled_data.skip(test_size)
    
    columns_to_remove = ["CONTENT", "CLASS", "COMMENT_ID", "AUTHOR", "DATE"]
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True,remove_columns=columns_to_remove)
    tokenized_test = test_dataset.map(preprocess_function, batched=True,remove_columns=columns_to_remove)

    model = AlbertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    
    training_args = TrainingArguments(
        output_dir="./Spam/results",
        fp16=True, 
        eval_strategy="steps", # model test itself every X steps
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        learning_rate=2e-5, # step size the model takes when correcting errors
        per_device_train_batch_size=8, # looks at 8 comments at a time before updating its weight
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01, # penalty for making weight too large
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        max_steps=1000,
        logging_steps=50
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )
    
    print("Start training ...")
    trainer.train()
    
    # runs the tokenized_test data through the model and return the accuracy
    eval_result = trainer.evaluate()
    print("\nTest Output:\n")
    print(f"Test Accuracy: {eval_result['eval_accuracy']:.2%}")

    trainer.save_model("./Spam/final_model")
    tokenizer.save_pretrained("./Spam/final_model")

if __name__ == "__main__":
    tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
    main()