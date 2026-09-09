# Temporal Concept Drift in Multi-label Legal Classification: A Comparative Benchmark of Encoder and Decoder Architectures

This repository contains the source code, datasets and experimental frameworks for the dissertation project evaluating neural architectures against multi-label complexity and temporal concept drift in United Kingdom and European Union legislation.

Unlike standard benchmark evaluations that introduce temporal data leakage via random splits, this project implements a strict chronological evaluation pipeline. It benchmarks fine-tuned domain-specific encoders against autoregressive Large Language Models (LLMs) and evaluates Spectral Decoupling as a robust regularisation strategy to protect rare legal concepts, following Chalkidis and Søgaard (2022).

## Project Overview

*   **Problem:** Legal text classification suffers from highly imbalanced taxonomies (overlapping categories, frequent head labels and rare tail labels) alongside temporal drift, where statutory language evolves over time.
*   **Approach:** Chronological data splitting (avoiding future data leakage) across historical UK and EU legislation datasets.
*   **Key Finding:** Fine-tuned legal encoders drastically outperform general-purpose LLMs in dense multi-label environments, offering superior Micro F1 and Macro F1 scores and a 50x reduction in inference latency at a lower computational cost. 
*   **Regularisation:** Incorporating Spectral Decoupling preserves the invariance of tail concepts, mitigating performance degradation over time for rare legal classes.
 
### Datasets Evaluated
*   **UK-LEX (UK-LEX-18):** Covers UK legislation from 1975 to 2018.
*   **EUR-LEX (EUR-LEX-21):** Covers European Union legislation from 1958 to 2016.

### Evaluated Architectures
*   **Fine-tuned Encoders:** `Legal-BERT` and `Legal-BERT-LWAN` (with and without Spectral Decoupling).
*   **Autoregressive Decoders:** `Llama 3 8B` evaluated via Zero-Shot and 3-Shot In-Context Learning (ICL).
