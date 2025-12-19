## ReadMe

fa25-google-dc/process-test-set.ipynb
- This file processes the test set by adding prompts to determine whether each prompt is harmful (1) or not harmful (0)

fa25-google-dc/process-outputs.ipynb
- This file processes the inference outputs to detemine if each output (0) responded to the prompt or (1) rejected the prompt.

All other csv and jsonl files left in the repository are outcomes of testing the scripts.


# Gemini Fine-Tuning Pipeline (Vertex AI)

This repository contains a complete data preparation, fine-tuning, and evaluation pipeline for **Gemini Supervised Fine-Tuning (SFT)** using **Vertex AI**. The pipeline covers dataset formatting, train/validation splitting, tuning job submission, and post-tuning evaluation.

---

## Overview

The pipeline performs the following steps:

1. Convert raw CSV datasets into Gemini-compatible JSONL  
2. Upload datasets to Google Cloud Storage (GCS)  
3. Split data into training (90%) and validation (10%)  
4. Launch a Vertex AI Gemini fine-tuning job  
5. Evaluate the tuned model on held-out validation data  
6. Upload evaluation results back to GCS  

---

## Project Configuration

### Google Cloud
- **Project ID**: `soy-surge-474318-q8`
- **Region**: `us-central1`
- **GCS Bucket**: `test_generic`

### GCS Directory Structure
gs://test_generic/
├── train/ # Training JSONL files (90%)
├── val/ # Validation JSONL files (10%)
└── results/ # Evaluation outputs

---

## Fine-Tuning Configuration

### Tuning Method
- **Type**: Supervised Fine-Tuning (SFT)
- **Platform**: Vertex AI Gemini tuning API

### Model Parameters
- **Base model**: `base_model` (specified externally at runtime)
- **Tuned model display name**: `audits_tuned_model_with_validation`
- **Epoch count**: `10`

> Learning rate, batch size, optimizer, and scheduler are managed internally by Vertex AI and are not user-configurable in the Gemini SFT API.

### Training Dataset
- **GCS URI**: `gs://test_generic/train/audits_train_formatted.jsonl`
- **Format**: JSONL (Gemini SFT schema)
- **Shuffle**: Enabled
- **Random seed**: `42`

### Validation Dataset
- **GCS URI**: `gs://test_generic/val/audits_train_formatted_val.jsonl`
- **Purpose**: Periodic evaluation during fine-tuning

---

## Dataset Split Parameters

- **Train / Validation split**: 90% / 10%
- **Shuffling**: Enabled before splitting
- **Validation filename suffix**: `_val.jsonl`
- **Training files**: Overwrite originals in `train/`

---

## Data Format Requirements

### Gemini SFT JSONL Schema

Each line in the dataset must follow this structure:

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{ "text": "<prompt>" }]
    },
    {
      "role": "model",
      "parts": [{ "text": "<expected_response>" }]
    }
  ]
}
