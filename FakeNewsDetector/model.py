import csv
import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
import numpy as np

MODEL = "bert-base-uncased"
TEST_SIZE = 0.4
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3

# prepare data so BERT can read it
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # run for every sample when training
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Remove batch dimension
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    """ Compute accuracy, precision, recall and f1_score"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def load_data(filename, fake=True):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        evidence = []
        labels = []
        
        for row in reader:
            # Format: Title, Text, Subject
            evidence.append([cell for cell in row[:3]])
            labels.append(1 if fake else 0)
            
    return evidence, labels

def format_text(data):
    """Format the text from [title, text, subject] to a single string"""
    return [
        f"Title: {item[0]}\nText: {item[1]}\nSubject: {item[2]}"
        for item in data
    ]

def main():
    dir_path = "FakeNewsDetector"
    
    # Load data
    print("Loading data...")
    evidence1, label1 = load_data(os.path.join(dir_path, "Fake.csv"), fake=True)
    evidence2, label2 = load_data(os.path.join(dir_path, "True.csv"), fake=False)
    
    # Combine data
    evidence = evidence1 + evidence2
    labels = label1 + label2
    
    print(f"Total samples: {len(labels)} (Fake: {sum(labels)}, True: {len(labels) - sum(labels)})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    
    # Format text
    X_train = format_text(X_train)
    X_test = format_text(X_test)
    
    # Load tokenizer
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    # Load model
    model = BertForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=2
    )
    
    # Create datasets
    train_dataset = NewsDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = NewsDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    # Training arguments
    # evaluate after each epoch
    # save the best model
    # optimize based on F1 score
    # learning rate warmup to reduce training spikes
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,  
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,  
        warmup_steps=500,  
    )
    
    # Create trainer
    # handles training loop, validation loop, metrics, saving model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating model...")
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(axis=1)
    
    print("\nCLASSIFICATION REPORT (Test Set)")
    print(classification_report(
        y_test, y_pred,
        target_names=['True News', 'Fake News']
    ))
    
    # Save model
    print("\nSaving model...")
    model.save_pretrained("./FakeNewsDetector/model")
    tokenizer.save_pretrained("./FakeNewsDetector/model")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()