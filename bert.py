import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle

# 1. Load your dataset
df = pd.read_csv("data.csv")  # columns: 'text', 'label'

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# 2. Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split train/test
train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
test_dataset = train_test["test"]

# 3. Load Model
num_labels = len(label_encoder.classes_)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# 4. Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 5. Training Arguments (compatible with transformers 4.55.0)
training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    logging_steps=100,
    save_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 7. Train
trainer.train()

# 8. Save Model, Tokenizer, and Label Encoder
model.save_pretrained("distilbert_doc_classifier")
tokenizer.save_pretrained("distilbert_doc_classifier")

with open("distilbert_doc_classifier/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# 9. Prediction Example
def predict(text, model, tokenizer, label_encoder):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = outputs.logits.argmax().item()
    return label_encoder.inverse_transform([predicted_class_id])[0]

# Load for inference example
loaded_model = DistilBertForSequenceClassification.from_pretrained("distilbert_doc_classifier")
loaded_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert_doc_classifier")
with open("distilbert_doc_classifier/label_encoder.pkl", "rb") as f:
    loaded_label_encoder = pickle.load(f)

print(predict("Patient reports high blood sugar and fatigue, scheduled for insulin therapy", loaded_model, loaded_tokenizer, loaded_label_encoder))
