# NS-FirmID: A Neuro-Symbolic Multi-Agent Framework for Reliable Firmware Version Identification at Internet Scale.

This repository contains the implementation and experimental artifacts for NS-FirmID, the first neuro-symbolic multi-agent framework tailored for reliable firmware version identification at Internet scale. By orchestrating neural perception with symbolic rigor, NS-FirmID effectively navigates the resource-precision trade-off and mitigates the hallucination-induced false positives prevalent in pure LLM-based identification.📂 Repository Structure & Module OverviewThe repository is organized into synergistic modules that mirror our "Extract-Verify-Decide" workflow

## 📂 Repository Structure & Module Overview

The repository is organized into synergistic modules that mirror our **"Extract-Verify-Decide"** workflow.

### 1. Extractor Agent (`Extractor_agent_construct/`)
Responsible for the neural semantic parsing of noisy banners, balancing reasoning capability with deployment efficiency.
* **`agent_sft_Qwen2.../`**: Contains the model weights for the student SLM (Qwen2.5-7B), fine-tuned via knowledge distillation from DeepSeek-R1
* **`query_llm_with_logprobs.py`**: An inference engine that extracts device attributes $(b, m, v)$ while capturing token-level log-probabilities for uncertainty modeling.
* **`prompt.json`**: Meticulously designed system prompts incorporating Chain-of-Thought (CoT) templates to regularize reasoning dynamics.

### 2. Validator Agent (`KB-device_database/` & `Knowledge_graph_construct/`)
Functions as the symbolic anchor, scrutinizing neural extractions against a structured knowledge base to defend against structural hallucinations.
* **`KB-device_database/`**: The hierarchical device knowledge base $\mathcal{K}$ aggregated from official CPE, Cydar, and custom vendor crawlers.
* **`result_verify.py`**: Implements the **Hierarchical Fuzzy Matching Algorithm** (Algorithm 1) to validate attribute alignment.
* **`similarity_calculate.py`**: Computes hybrid similarity metrics, combining string sequence matching with semantic profiling for precise model identification.

### 3. Discriminator Agent (`discriminator_construct/`)
A gray-box reliability estimator that fuses neuro-symbolic evidence into a 28-dimensional feature vector to rigorously quantify confidence.
* **`analysis_token_logprobs.py`**: Analyzes the EA's inference trace to compute intrinsic neural uncertainty metrics, including Perplexity (PPL) and Low-Confidence Ratio.
* **`confidence_discriminator.py`**: Detects the "Calibration Gap" between the model's claimed confidence and its objective generation probability.
* **`discriminator_model_optimized...`**: The trained XGBoost classifier used to accept or safely abort extractions based on the confidence threshold $T_{c}$.

### 4. Vulnerability Mapping (`vulnerability_match/`)
The downstream application module that transforms raw asset discovery into actionable risk intelligence.
* **`vul_match_client.py`**: Systematically transforms extracted attributes into standardized CPE URIs to establish reliable linkages to the National Vulnerability Database (NVD).

---

## 🛠️ Performance & Scalability
NS-FirmID is optimized for high-throughput Internet-scale auditing.
* **Accuracy**: Sustains a 93% F1-score and reduces the false positive rate (FPR) to 4.1%.
* **Efficiency**: Operates >60x faster than massive teacher-model APIs, achieving a throughput of ~10,400 samples/hour on a single NVIDIA A100 GPU.
* **Robustness**: Maintains stable performance across diverse banner heterogeneity tiers and remains resilient under high adversarial interference.


## ⚖️ Ethical Statement & Usage Policy

This research and the associated NS-FirmID framework are intended strictly for academic exchange and cybersecurity research purposes. By accessing or utilizing the artifacts in this repository, users must commit to the following ethical and legal obligations:

* **Academic Use Only**: This project is designed to improve global network security awareness and facilitate the understanding of IoT ecosystem risks. It must not be used for any unauthorized or malicious activities.

* **Compliance with Ethics**: Users are required to strictly adhere to the highest ethical standards and local legal regulations governing internet measurement and data privacy.
* **Access Authorization**: To prevent potential misuse, the active scanning framework for device discovery and the high-precision vulnerability matching engine are not open for unrestricted use. Access to these specific modules requires formal academic authorization.
* **Contact for Authorization**: For inquiries regarding research collaboration or access to the scanning system, please contact the authors at 1441687322@qq.com.
