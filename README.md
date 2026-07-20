# Project Instructions

This repository provides the implementation, datasets, and supplementary materials for paper: A Causal Perspective on the Role of Training Dynamics for Interpreting Privacy Risks in Code Models

## Additional Tables
Supplementary tables are available in **`./extra_tables/`** for reference and reproducibility.  

## Environment Setup
1. Create a Python environment (tested with **Python 3.12**).  
2. Install the required dependencies (adjusting the CUDA version as appropriate):  
   ```bash
   python -m pip install -r requirement.txt
   ```  
Each model is configured through its corresponding `.yaml` file under **`./configs/`**. For illustration, we highlight the configuration for the **StableCode-3B** model.  

## Dataset
We provide the **PII dataset** used in the paper under **`./dataset_all/`**. To construct a customized dataset using our pipeline, modify the configuration file and run:  
```bash
python run.py --config configs/stabilityai_stable-code-3b_fim.yaml --task dataset
```  

### Detection Performance (Recall & Precision)
The human-annotated ground-truth evaluation of the PII retrieval stage (paper **Section 5.2, Table 3**) is provided under **`./detection_eval/`**. The script reproduces the per-type recall/precision of regex, StarPII, and their combination over the ground-truth set (**808 files, 1,689 exhaustively labeled PII spans** across six types):
```bash
python detection_eval/run_recall_precision.py
```
Because the ground-truth set contains real credentials and PII, the full file is **not shipped in this public repository** — only a redacted schema sample (`detection_eval/data/sample_redacted.jsonl`) is included. See **`./detection_eval/README.md`** for the record schema, how to obtain and place the full data file, environment requirements (a HuggingFace token for the gated `bigcode/starpii` model), and a description of the generated outputs.

## Fine-Tuning
Model fine-tuning can be performed with the following command:  
```bash
python run.py --config configs/stabilityai_stable-code-3b_fim.yaml --task finetune
```  

## Evaluation
**RQ1 Learning Difficulty Across PII Types:**  
```bash
python run.py --config configs/stabilityai_stable-code-3b_fim.yaml --task post_eval
```

**RQ2 Learning Difficulty–Leakage Risk Relationship:**  
```bash
python run.py --config configs/stabilityai_stable-code-3b_fim.yaml --task post_attack
```
**RQ3 Causal Effect of Learning Dynamics on Leakage Risk:**

We provide the scm process at ./causal/causal_notebooks

