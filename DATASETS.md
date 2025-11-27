# 📊 Consolidated Data Card & Acknowledgments

This document outlines the metadata, legal compliance, risk assessment, and usage details for all external datasets used in this cross‑modal analysis suite. Information is grouped by category to facilitate quick comparison and auditing.

---

## 1. Dataset Summaries

Overview of the datasets, their scale, and their original intended purpose.

| Dataset       | Type                     | Size                          | Original Motivation                                               | Primary Tasks                                |
|---------------|--------------------------|-------------------------------|-------------------------------------------------------------------|----------------------------------------------|
| **MS COCO**   | Vision & Language        | 330K images; 2.5M instances   | Advance vision research with realistic, contextual images.        | Object Detection, Segmentation, Captioning   |
| **CIFAR-10**  | Vision (32×32)           | 60,000 images                 | Benchmark for small-scale deep learning models.                   | Image Classification (10 classes)            |
| **Visual Genome** | Vision & Structured Language | 108K images; 5.4M descriptions | Connect structured linguistic knowledge to visual imagery.        | VQA, Scene Graph Generation, Phrase Grounding |

---

## 2. Legal and Ethical Considerations

Mandatory Attribution: When publishing work using this suite, the citations below must be included.

| Dataset       | License     | Required Citation (Paper) | Source             | Known Limitations / Biases                                   |
|---------------|-------------|---------------------------|--------------------|--------------------------------------------------------------|
| **MS COCO**   | CC BY 4.0   | Lin et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV. | [cocodataset.org](https://cocodataset.org) | Western-centric object distribution; subjective annotator bias |
| **CIFAR-10**  | MIT License | Krizhevsky (2009). *Learning Multiple Layers of Features...* | [cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html) | Low resolution; lacks real-world diversity                     |
| **Visual Genome** | CC BY 4.0 | Krishna et al. (2017). *Visual Genome...* IJCV. | [visualgenome.org](https://visualgenome.org) | Inherits biases from Flickr/COCO; annotation consistency varies |

> **Note:** For MS COCO and Visual Genome, images are sourced from Flickr and are subject to their original copyright licenses.

---

## 3. Project Risks & Mitigation

Specific risks identified for this project and the steps taken to address them.

| Dataset       | Identified Risk                                         | Mitigation Strategy in This Suite                                                                 |
|---------------|---------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **MS COCO**   | Geographic Bias: Models may underperform on non-Western scenes. | Evaluate performance on a dedicated "out-of-distribution" test set; do not claim global generalization. |
| **MS COCO**   | Annotator Subjectivity: Captions may reflect worker biases. | Use all 5 available captions per image during training to average out individual linguistic quirks. |
| **CIFAR-10**  | Oversimplification: Results may not scale to complex images. | Use strictly for "Proof of Concept" (PoC) and initial debugging; do not extrapolate findings to high-res tasks. |
| **Visual Genome** | Noisy Annotations: Short/vague descriptions (e.g., "blue sky"). | Pre-filtering: Discard region descriptions shorter than 5 words to ensure rich grounding signals. |

---

## 4. Project Use

How each dataset is specifically processed and utilized within this codebase.

| Dataset       | Usage Description                                                                 |
|---------------|-----------------------------------------------------------------------------------|
| **MS COCO**   | We use the 2017 Train/Val splits. Raw images and 5 captions are used for training cross-modal alignment encoders. |
| **CIFAR-10**  | Used for initial direct logit attribution and cosine similarity calculations regarding alignment. |
| **Visual Genome** | We extract region descriptions and relationship triples (Subject–Predicate–Object) to train scene understanding and measure alignment. |

---
