import os
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('project_dir', type=str, help='Path to the project directory')
args = parser.parse_args()

# Set the project directory from the command-line argument
project_dir = args.project_dir


# Load CSV files
train_df = pd.read_csv(os.path.join(project_dir, "train.csv"))
dev_df = pd.read_csv(os.path.join(project_dir, "dev.csv"))
test_df = pd.read_csv(os.path.join(project_dir, "test.csv"))

# Rename 'det_fake_label' column to 'label' for consistency with Hugging Face trainer
train_df = train_df.rename(columns={"det_fake_label": "label"})
dev_df = dev_df.rename(columns={"det_fake_label": "label"})
test_df = test_df.rename(columns={"det_fake_label": "label"})

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine into a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": dev_dataset,
    "test": test_dataset
})

# Load the tokenizer and model
model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["content"], padding="max_length", truncation=True, max_length=256)

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_datasets.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(project_dir, "results"),
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir=os.path.join(project_dir, "logs"),
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=len(tokenized_datasets["train"]) // (32 * 2),
    save_strategy="steps",
    save_steps=len(tokenized_datasets["train"]) // (32 * 2),
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,
    gradient_accumulation_steps=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on test set
test_results = trainer.predict(tokenized_datasets["test"])
print(test_results.metrics)

# Confusion Matrix
y_true = test_results.label_ids
y_pred = test_results.predictions.argmax(-1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix on Test Set")
plt.show()

# Metric Visualizations (Accuracy, Precision, Recall, F1)
metrics = test_results.metrics

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=list(metrics.keys())[:-1], y=list(metrics.values())[:-1], palette="viridis")
plt.title("Evaluation Metrics on Test Set")
plt.ylabel("Score")
plt.ylim(0, 1)
for i, v in enumerate(list(metrics.values())[:-1]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, test_results.predictions[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, test_results.predictions[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()
plt.show()
