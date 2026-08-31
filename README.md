# Health Misinformation Detection on the Med-MMHL Dataset

Automated detection of health-related misinformation using a fine-tuned **Bio_ClinicalBERT** transformer, benchmarked against classical ML baselines (SVM, Naive Bayes, Random Forest, Logistic Regression) and a Graph Neural Network, evaluated on the **Med-MMHL** article-level dataset.

## Overview

Digital health misinformation spread rapidly during the COVID-19 pandemic, and manual fact-checking can't keep pace with the volume of content produced. This project builds a binary classifier (real vs. fake) for health news articles by fine-tuning a domain-specific BERT variant pretrained on clinical text, and compares it against traditional machine learning baselines and a graph-based deep learning approach.

Full methodology, related work, and results are written up in `research_paper-MEDMHL.docx`.

## Repository Contents

| File | Description |
|---|---|
| `train_fakenews_model.py` | Main training script: fine-tunes `emilyalsentzer/Bio_ClinicalBERT` on the article dataset and produces evaluation plots |
| `BIOBESRTFINAL.ipynb` | Notebook version of model development/experimentation (BioBERT + ClinicalBERT) |
| `BIOBESRTFINAL-checkpoint.ipynb` | Autosaved checkpoint of the notebook above |
| `train.csv` / `dev.csv` / `test.csv` | Article-level train / validation / test splits |
| `dev-checkpoint.csv` | Duplicate/backup copy of `dev.csv` |
| `fakenews_article.zip` | Archive containing both the article-level (`fakenews_article/`) and tweet-level (`fakenews_tweet/`) Med-MMHL splits |
| `research_paper-MEDMHL.docx` | Full write-up: abstract, literature review, methodology, and results |
| `researchpaper.docx` | Separate short paper on the mathematical foundations of NLP (not part of the model pipeline) |

## Dataset

Each CSV has three columns: an index column, `content` (article text), and `det_fake_label` (`0` = real, `1` = fake).

| Split | Rows | Real (0) | Fake (1) |
|---|---|---|---|
| train | 8,309 | 4,017 | 4,292 |
| dev | 1,189 | 575 | 614 |
| test | 2,375 | 1,148 | 1,227 |

Labels are fairly balanced across all three splits. The `fakenews_article.zip` archive additionally contains a `fakenews_tweet/` subfolder with shorter, tweet-level splits of the same dataset, intended for the GNN experiments described in the paper.

## Model

- **Architecture:** `AutoModelForSequenceClassification` on top of `emilyalsentzer/Bio_ClinicalBERT` (2 output labels)
- **Tokenization:** max length 256, padded/truncated
- **Training:** learning rate `2e-5`, batch size 32 (train/eval), 2 epochs, weight decay `0.01`, gradient accumulation over 2 steps, mixed precision (`fp16`)
- **Model selection:** best checkpoint chosen by validation **F1**, evaluated/saved every `len(train) // 64` steps
- **Evaluation metrics:** accuracy, precision, recall, F1, plus confusion matrix, ROC curve (AUC), and precision-recall curve on the held-out test set

## Requirements

```bash
pip install pandas torch datasets scikit-learn transformers matplotlib seaborn wordcloud
```

A CUDA-capable GPU is strongly recommended (the script uses `fp16` mixed-precision training).

## Usage

1. Unzip the dataset (or use the provided `train.csv` / `dev.csv` / `test.csv` directly) so that `train.csv`, `dev.csv`, and `test.csv` live in the same project directory.
2. Run the training script, passing the path to that directory:

```bash
python train_fakenews_model.py /path/to/project_dir
```

The script will:
1. Load and rename the label column (`det_fake_label` → `label`)
2. Tokenize all three splits with the Bio_ClinicalBERT tokenizer
3. Fine-tune the model with Hugging Face `Trainer`
4. Evaluate on the test set and print the metrics dictionary
5. Display a confusion matrix, a bar chart of accuracy/precision/recall/F1, an ROC curve, and a precision-recall curve

Checkpoints and logs are written to `project_dir/results` and `project_dir/logs`, respectively.

## Notes

- `BIOBESRTFINAL.ipynb` reflects earlier, exploratory experimentation (including an attempt with vanilla `bert-base-uncased` and `dmis-lab/biobert-v1.1`) and contains hardcoded local Windows paths (e.g. `C:\Users\Harshita\...`) that will need to be updated before re-running it elsewhere.
- The `-checkpoint` files (`BIOBESRTFINAL-checkpoint.ipynb`, `dev-checkpoint.csv`) are Jupyter/editor autosave artifacts and can generally be ignored or removed once the main files are confirmed up to date.
- `researchpaper.docx` (Mathematical Foundations of NLP) is unrelated to the fake-news pipeline and appears to be a separate assignment.
