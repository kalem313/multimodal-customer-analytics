# Multimodal Customer Analytics

This project implements an end-to-end pipeline for structuring messy, unlabeled image-caption data. It leverages CLIP for multimodal embeddings and BERTopic for unsupervised clustering to automatically generate labels for business analytics.

## Key Features
* **Data Extraction:** Automated downloading and validation of the TextCaps dataset.
* **Multimodal Embedding:** Maps images and text to a shared vector space using OpenAI's `clip-vit-base-patch32`.
* **Unsupervised Clustering:** Uses BERTopic to discover hidden themes (e.g., "Street Signs," "Computer Hardware") without manual labeling.
* **Supervised Distillation:** Trains a Linear SVM on the discovered clusters to create a lightweight inference engine (93% Accuracy).

## Tech Stack
* Python 3.10+
* BERTopic (Clustering)
* HuggingFace Transformers (CLIP Model)
* Scikit-Learn (SVM Classification)

## Results
* **Clusters Found:** 17 distinct visual/textual topics.
* **Classification Accuracy:** 93% on the hold-out test set.

## How to Run
1. Clone the repo.
2. Install dependencies
3. Run the Notebook.
