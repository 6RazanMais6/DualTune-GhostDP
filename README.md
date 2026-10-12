# DualTune-GhostDP

### A Unified Approach for Synergistic Differentially Private Fine-Tuning of Language Models

**Official experimental code repository accompanying our TMLR paper.**

**Authors:** Razan El Mais, Ali Chehab, Ibrahim Issa, and Razane Tajeddine  
**Affiliation:** Department of Electrical and Computer Engineering, American University of Beirut, Lebanon  
**Venue:** Transactions on Machine Learning Research (TMLR), 2026  
**Paper:** https://openreview.net/forum?id=aJ74dIAQZH

## Overview

DualTune-GhostDP investigates a two-phase fine-tuning approach that combines sanitized-data adaptation, differentially private optimization, Ghost Clipping, and Edgeworth-based privacy accounting to improve the privacy–utility–memory trade-off of language model fine-tuning.

The experimental evaluation covers encoder-only transformer models (RoBERTa and BERT) and a decoder-only extension (DistilGPT-2 and GPT-2).

## Repository Organization

This repository contains the experimental Python scripts and Google Colab notebooks used in the study.

### 1. Encoder Models — Main Experiments

**Directory:** `Encoder_Models/RoBERTa_BERT/`

The main experiments are conducted on three GLUE benchmark tasks:

- **SST-2:** Sentiment classification
- **QNLI:** Question-answering natural language inference
- **QQP:** Quora Question Pairs classification

The experiments are organized as follows:

| Experiment | Description |
|---|---|
| Experiment 1 | Conventional single-phase DP-SGD with standard clipping |
| Experiment 2 | Two-phase fine-tuning with standard clipping |
| Experiment 3 | Proposed two-phase DualTune-GhostDP configuration |

The released scripts use RoBERTa-base as their active encoder backbone. BERT-base experiments follow the corresponding architecture-adapted implementations using `bert-base-uncased`, `BertTokenizer`, and `BertForSequenceClassification`.

### 2. Decoder-Only Models — Generalization Extension

**Directories:** `Decoder_Models/DistilGPT2/` and `Decoder_Models/GPT2/`

This extension evaluates DualTune-GhostDP on DistilGPT-2 and GPT-2 using the SST-2 dataset.

Three configurations are included:

| Configuration | Description |
|---|---|
| D1 | Conventional DP-SGD with Weight Tying and standard clipping |
| D2 | Single-phase DP-SGD without Weight Tying, using Ghost Clipping |
| D3 | Two-phase DualTune-GhostDP without Weight Tying, using Ghost Clipping |

Each configuration is provided as a Google Colab notebook. The notebooks include environment setup, experimental training code, seed configuration, evaluation, and logging procedures.

## Code Availability and Execution

The repository provides the experimental scripts and notebooks associated with the paper.

**Encoder-only experiments:** The Python scripts use PyTorch, Hugging Face Transformers and Datasets, spaCy, and `private-transformers`. Experiments require appropriate Python dependencies and access to the GLUE datasets.

**Decoder-only experiments:** The Jupyter notebooks were developed for Google Colab with GPU acceleration. They configure a dedicated Python environment and use Google Drive for saving experiment outputs.

To execute an experiment, inspect its configuration, dependencies, dataset preprocessing, privacy settings, and output paths before running.

The released files represent the research implementations. Some configurations and environment-specific details may require adaptation to reproduce individual manuscript results.

## Privacy Considerations

The approach combines a nonprivate adaptation phase on deterministically sanitized text with a subsequent differentially private training phase. The formal scope of the privacy guarantees and associated assumptions should be interpreted according to the manuscript.

## Citation

If you use the code, please cite:

El Mais, R., Chehab, A., Issa, I., and Tajeddine, R. (2026). *DualTune-GhostDP: A Unified approach for Synergistic Differentially Private Fine-Tuning of Language Models*. Transactions on Machine Learning Research.

**OpenReview:** https://openreview.net/forum?id=aJ74dIAQZH

## Related Publication

El Mais, R., Chehab, A., Issa, I., and Tajeddine, R. (2026). *Is Weight Tying Still Beneficial for Decoder-Only LLMs in Private Settings Under DP-SGD?* Accepted at IEEE TPS 2026.

Preprint: https://arxiv.org/abs/2609.40335

## Contact

For questions regarding the implementation, please contact the corresponding authors through their institutional affiliations.
