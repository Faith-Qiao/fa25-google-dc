# Constitutional AI Fine-Tuning Pipeline (Vertex AI)

This repository contains the SAAS team's **Constitutional AI (CAI)** pipeline implementation. Using the framework established in Anthropic's research, we implemented and compared five different "Constitutions" to evaluate their effectiveness in mitigating harmful responses to red-team prompts.

## Project Overview

We compared five distinct constitutions against a control (default Gemini 2.5-flash-lite) to determine which set of principles best steers model behavior toward safety without sacrificing utility.

**The Five Constitutions:**

1. **Generic**
2. **CSRM**
3. **Environmental, Health, and Safety (EHS)**:
4. **Audits**
5. **Security**

---

## Repository Structure

```text
fa25-google-dc/
├── original_datasets/           # Raw red-team training and test sets
├── vertex_ai_scripts/           # Scripts to run Critique/Revision on Vertex AI
├── fine_tuning_datasets/        # Final revised prompt-response pairs per constitution
├── gemini_fine_tuning.ipynb     # Main notebook for supervised fine-tuning (SFT) jobs
├── original_inference_outputs/  # Batch inference results from all 6 models
├── process-test-set.ipynb       # Labeling script for Ground Truth
├── process-outputs.ipynb        # Labeling script for model rejections
└── README.md

```

---

## The Constitutional AI Workflow

### 1. Data Preparation

We compiled a dataset of red-team prompts, split into `train_full.csv` and `test_full.csv`. These are located in `original_datasets/`.

### 2. Critique & Revision Pipeline

Following Anthropic's original paper, we ran the training set through a three-step pipeline:

1. **Initial Answer**: Model responds to red-team prompt.
2. **Critique**: Model evaluates its response against a specific Constitution.
3. **Revision**: Model rewrites the response to adhere to the Constitution.

Because of the dataset's size (thousands of prompts), we utilized **Vertex AI** for this stage.

* **How to run**: Upload `vertex_ai_scripts/` to the GCP Cloud Terminal.
* **Execution**: Run `python run_pipeline.py`.
* **Configuration**: Modify the `# ----- CONFIG -----` block in `run_pipeline.py` to match your Project ID, Bucket, and desired Constitution.

### 3. Supervised Fine-Tuning (SFT)

Once we generated the final revised responses (found in `fine_tuning_datasets/`), we fine-tuned **Gemini 2.5-flash-lite** five separate times—one for each constitution.

* The fine-tuning logic is handled in `gemini_fine_tuning.ipynb`.
* This converts the revised pairs into the required Gemini SFT JSONL format.

### 4. Evaluation & Inference

We ran the `test_full.jsonl` through 6 models: the 5 fine-tuned versions and 1 default Gemini (control).

* Inference was performed using the **Vertex AI Batch Inference UI**.
* The results are stored in `original_inference_outputs/`.

---

## Analysis & Labeling

To evaluate the effectiveness of each constitution, we use an LLM-as-a-judge approach to label our data. This requires a two-step process for both the test prompts (Ground Truth) and the model responses (Inference).

### 1. Preparing the Grading Prompts

We use two notebooks to format our data into "grading requests" for the model:

* **`process-test-set.ipynb`**: Takes the raw test set and appends a classification prompt to each entry. This asks the model to output `1` if the user prompt is harmful or `0` if it is harmless.
* **`process-outputs.ipynb`**: Takes the 6 inference outputs and appends a classification prompt. This asks the model to output `1` if the fine-tuned model successfully rejected the prompt or `0` if it answered it.

### 2. Executing Batch Inference (Labeling)

Once the notebooks have generated these "grading" JSONL files, we run them through the **Vertex AI Batch Inference UI**:

1. **Upload** the processed JSONL files to GCS.
2. **Select** the base Gemini 2.5-flash-lite model to act as the judge.
3. **Run Batch Inference** to get the actual labels (the 0s and 1s).

### 3. Final Comparison

The results of these labeling runs and confusion matrices are stored in the separate `saas-google-ml-analysis` repository. By comparing the **Ground Truth** (is the prompt actually bad?) against the **Model Output** (did the model reject it?), we generate **Confusion Matrices** for each of the 5 constitutions and the control model to measure safety precision and recall.

---

## Technical Configuration (Vertex AI)

### Google Cloud Environment

* **Project ID**: `your-project-id`
* **Region**: `us-central1`
* **GCS Bucket**: `gs://attack_prompts`
* **Base Model**: `gemini-2.5-flash-lite`

### Fine-Tuning Parameters

* **Tuning Method**: Supervised Fine-Tuning (SFT)
* **Train / Validation Split**: 90% / 10%
* **Epochs**: 10
* **Random Seed**: 42

### Gemini SFT JSONL Schema

Each line in the fine-tuning datasets follows this structure:

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{ "text": "<red_team_prompt>" }]
    },
    {
      "role": "model",
      "parts": [{ "text": "<final_revised_response>" }]
    }
  ]
}
