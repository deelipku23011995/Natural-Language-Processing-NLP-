# Natural-Language-Processing-NLP-

## Domain-Specific Fine-Tuning of Encoder Models (RoBERTa / BERT)
This repository provides an end-to-end NLP pipeline for fine-tuning encoder-based Transformer models (such as RoBERTa and BERT) on domain-specific classification and extraction tasks. It combines SpaCy for high-performance linguistic preprocessing with the Hugging Face Trainer API and Optuna for automated hyperparameter tuning.

Key Features

Transformer Fine-Tuning (Hugging Face): Leverage pre-trained models (roberta-base, bert-base-uncased, domain-adapted variants  with custom heads.

Automated Hyperparameter Tuning:  to optimize learning rates, batch sizes, weight decay, and warmup steps.

Standardized Evaluation Metrics: Built-in evaluation routines computing Precision, Recall, F1-Score, and Accuracy using evaluate/scikit-learn.

Export & Inference: Modular scripts to save tuned checkpoints and run inference on unseen text.
