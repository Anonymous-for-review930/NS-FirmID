We thank the reviewers for their constructive feedback and for recognizing our work's significance. We've uploaded a revised manuscript addressing all reviewer concerns in the attachment (all modifications highlighted in blue); core concerns are addressed below:

# 1. Systemic-Novelty & Neuro-Symbolic-Synergy (R29D, R29B)
R29D: Our novelty lies in a reliability-oriented neuro-symbolic security-architecture, rather than developing new ML-primitives. Pure LLMs (even 671B-Teacher) suffer unacceptable FPRs (16.9%-21.2%) in security-measurement due to hallucinations. We solve this reliability-bottleneck via a zero-trust pipeline where VA and DA act as fundamental error-control mechanisms, not merely incremental-patches:

- Gray-Box Uncertainty-Modeling (DA): Fuses internal neural-signals (token-log-probabilities), Chain-of-Thought semantic-stability, and external symbolic-evidence into a 28-dimensional hallucination-signature, dropping FPR to 4.1%.
- Zero-Retraining Knowledge-Injection (R29B): VA dynamically integrates historical-versions into the EA’s prompt via In-Context-Learning. This induces EA to learn vendor-specific naming-semantics for secondary-extraction without expensive model-retraining.

# 2. Dataset Scale, Generalization & KB-Dependency (R29D, R29A, R29E)
- Scale & Generalization (R29D): Fine-grained IoT version-identification lacks public datasets. Our 5,185 labeled-samples are strictly deduplicated, span >1,000 brands (Table-II). Our in-the-wild deployment achieved coverage 211x broader than Nmap across 150,000 unverified-banners. For furture work, we plan to open-source an expanded annotated-dataset (>20,000 entries).
- KB-Dependency (R29A-Q3): Despite immense initial manual-effort to bootstrap the knowledge-base, our crawling-pipeline is now fully-automated and integrated into our lab's device-search-engine, reducing future-maintenance to minimal script-tweaking for entirely novel-vendors.
- Ethics (R29E-Q2): Individually notifying 44k device-owners is operationally-infeasible and legally-complex. Following measurement-ethics, we will strictly anonymize the released-dataset and have reported critical-vulnerabilities to the top-5 most-affected vendors. While code is open-sourced, accessing our active scanning-framework for device-discovery requires strict academic-authorization.

# 3. Heterogeneity Quantification & Empirical-Correlation (R29C)
Heterogeneity-Correlation (R29C-Q1): To rigorously quantify banner-chaos, we developed a 5-dimensional weighted-scoring-scheme (integrating Shannon-Entropy, length, word-count, digit-ratio, and structure-score) to classify datasets into Low, Medium, and High heterogeneity-tiers. Results (Sec.IV-H, Fig.11) demonstrate NS-FirmID sustains remarkably stable-accuracies across all tiers: 92.5% (Low), 95.0% (Medium), and 92.3% (High). Our Pearson-correlation-heatmap further proves that while structural-complexity strongly-increases processing-time (r=0.934), its correlation with identification-accuracy remains very low. This validates our architecture's resilience in absorbing extreme semantic-noise without performance-degradation.

# 4. Adversarial-Robustness & Spoofing-Mitigation (R29D, R29A)
4.1. Conflict-Resolution: To resolve multi-version chaos, we use a three-stage zero-trust pipeline:

- Structural-Separation (EA): Domain-specific CoT forces EA to parse component-versions into distinct JSON-fields, preventing entity-confusion.
- Pattern-Verification (VA): Confounding-versions typically violate firmware-naming conventions, triggering deterministic-rejection and ICL-refinement.
- Ambiguity-Filtering (DA): Semantic-conflicts degrade token-probabilities; DA detects this neuro-symbolic uncertainty to safely-abort.

4.2. Spoofing-Mitigation:

- Format-Spoofing: When substitute versions with invalid-formats, VA rejects and retrieves historical-patterns to guide EA. Recognizing pattern-inconsistencies, EA generates uncertainty and lowers extraction-confidence, helping DA filter the spoofed-version.
- Semantic-Spoofing: Perfectly replacing version with valid historical/future formats remains indistinguishable at the textual layer; we document this limitation as an inherent visibility-constraint shared by banner-based systems.
- Decoy-Injection: Confounding-strings trigger semantic-conflicts. DA can detect underlying token-probability-degradation to prevent false-positives. Interference-evaluation (Fig.12) proves sustained 93%~95% resilient-accuracy under high-interference (up to 30 version-like strings), declining to 78.18% only under extreme-obfuscation, proves DA safely-aborts ambiguous-extractions (outputting null).

# 5. Inference-Efficiency & Deployment-Cost (R29A, R29B, R29E)
- Efficiency & Throughput (Table-V): Nmap processes rapidly (7.88ms) but yields 0.001 F1-score on heterogeneous banners. NS-FirmID trades latency for high-fidelity, achieving a 0.930 F1-score at 345.54ms/sample (~10,400 samples/hour on one A100-GPU), outpacing general SLMs (Llama-3-8B: 571.98ms) and operates >60x faster than the 671B-Teacher-API (21.5s/sample), entirely eliminating privacy-risks and token-costs.
- Regex-Generation & Hybrid-Deployment (R29E-Q1): VA-validated high-confidence patterns can automatically-compile into regular-expressions to feed fast first-stage-scanners (e.g., Nmap). NS-FirmID then optimally-serves as a second-stage analyzer, dedicated to large-scale, complex, long-tail assets where traditional rule-based tools inherently-fail.

# 6. Precision-Recall Trade-off & CVE-Mapping Impact (R29A, R29B)
- Filtered-IDs (R29A-Q4): DA fusing LLM-reasoning, internal token-probabilities, and external symbolic-validation to assess overall-confidence. By assigning low confidence-scores, it can filter extractions lacking contextual-evidence (e.g., isolated ultra-short authentic-versions, multiple identically-formatted plausible-candidates) or exhibiting anomalous-formats (e.g., anomalous-suffixes like 6.5.4.15-116n.jpn, or chaotic-mixtures like w.ar934x.v5.5.10_tycon_unitb.24238.141001.1641). We trade a marginal 7.3% long-tail recall-loss to halve FPR, preventing alert-fatigue.
- Precise-Version & CVE-Mapping Impact (R29B-Q1, R29A): Identifying only "Cisco-Router" yields massive false-positives in risk-assessment. Precise firmware-versions uniquely distinguish actively-exploitable deployments from patched-ones. By systematically transforming raw text into standardized CPEs and matching against the NVD, NS-FirmID reliably mapped 44,13 high risk CVEs (Sec.IV-F), effectively elevating raw-discovery into actionable, high-resolution risk-intelligence.

# 7. Errata (For All Reviews)
We corrected all R29E-identified typographical-errors (including algorithm-indentation, numerical-precision, and B&W-printability of figures).

Key updates in revised paper:
(1) Fig.1 (banner examples) and Fig.3 (optimized dataset).

(2) Heterogeneity scoring and correlation analysis (Sec.IV-H, Fig.11).

(3) Robustness evaluations against adversarial interference (Fig.12).

(4) Table-V (Sec.IV-G) for end-to-end efficiency/throughput benchmarks.
