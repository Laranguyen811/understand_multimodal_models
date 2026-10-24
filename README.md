
# Understanding Multimodal Models and Detecting Misalignment Research Direction
My formed agenda involves understanding multimodal models and detecting misalignment. Details include:

* Designing experiments to understand multimodal models using various techniques, including mechanistic interpretability and representation engineering
* Improving on the open-source suite to help the community verify and extend what I have done
* Building tools to detect misalignment, including undesirable behaviours, for current and future models

Since 2025, I have been working on understanding multimodal models by conducting experiments and exploring methods for detecting misalignment. Misalignment in AI refers to artificial intelligence systems that fail to align with human values, goals, or intentions. My experiments have shown some results.

What are these results? One experiment on the COCO dataset (Common Object in Context dataset by Microsoft) reveals critical gaps between true semantic understanding and benchmarking. True semantic understanding refers to a genuine grasp of meaning, and benchmarking means comparing against a standard. It shows 0.216 average cosine similarity (a measure between two vectors) for matching text-image pairs versus near-zero for mismatching ones.

Besides, another experiment involves a compositional copying (giving a model a text cue and seeing what visual feature it copies) task using the Visual Genome (a dataset containing relationships between objects and scenes) dataset. It shows confident predictions on brittle features unlikely to survive distribution shifts (data that shift away from the training data) in the real world. an 86.8% probability score (on a scale of 10, its confidence level is roughly 9). However, it relies on tiny embedding (numerical vector representing objects) differences (0.019 cosine similarity and 0.074 patch-level probability).

First, as we have observed over the last few years, AI has moved so quickly, and we do not have reliable ways of predicting its real-world performance. The International AI Safety Report 2026 states that general-purpose AI capabilities have continued to improve, even when their capabilities are jagged, meaning that they excel in some tasks while failing at some other simple tasks. Benchmarks often fail to predict real-world performance since many models have been trained using data from these same benchmarks (data contamination). It leads to inflated scores that do not reflect a model’s genuine ability. We need radically different approaches to understanding these models and their capabilities reliably.

Additionally, our understanding of models and the capabilities to detect misalignment in models, especially multimodal models, lags. Balasubramanian et al (2025) assert that mechanistic interpretability lags behind for multimodal models compared to LLMs, although multimodal models are the future. What is mechanistic interpretability? It is a technique of finding a specific algorithm or computation that a model learns during training.

On the other hand, Dan Hendrycks and Laura Hiscott, in their article “The Misguided Quest for Mechanistic AI Interpretability”, warn us about high investments in mechanistic interpretability without corresponding returns. Their central arguments advise against investing too much in ideas unlikely to work, potentially neglecting more effective ones. We should be more sceptical of giving mechanistic interpretability too many resources at the expense of other types of AI safety research.

In my research, I have observed that mechanistic interpretability can work in some experiments with certain changes that I have applied. Applying mechanistic interpretability in its current approach without looking at the bigger picture will likely fail. The current approach usually involves studying a small circuit (a subgraph of neural networks) with a toy model (a small, constructed model). Therefore, we need to combine different approaches, including representation engineering, holistically.

Furthermore, research alone does not account for mission-level impact in complex systems (Belcher B, Bonaiuti E. and Thiele G., 2024). We need to integrate it into social processes, including but not limited to policymaking, community building and businesses. We have to genuinely engage and involve stakeholders in our research and understand the intended users’ needs. By working with various stakeholders and receiving their feedback, I have purposefully organised my research around them. And, I will continue doing so with the same discipline and fervour.

In conclusion, my work directly contributes to the understanding of models and capabilities, with experiments to deepen our knowledge of multimodal models and how to detect misalignment in multimodal models. Beyond that, I seek to create a dialogue with people who will use my research. It is a small, meaningful contribution to preventing catastrophic risks from AI. It is a different approach from previous research, showing a shift away from well-trodden paths. According to the universal hypothesis, the results from my work will likely be useful for our endeavours with future powerful models.

## 🔍 Project Overview
Including but not limited to:
- Investigates how mechanistic interpretability techniques can be applied to cross-modal understanding.
- Supports modular experimentation across modalities (e.g., image, text, audio).

## 🧭 Branching Philosophy

This repository uses `dev` as the **default branch** to reflect its active, research-oriented development. Stable releases and polished modules will be merged into `main` when ready.

- `dev`: Active development, experimental features, evolving structure.
- `main`: Reserved for stable, documented releases.

## 🛠️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Laranguyen811/cross_modal_analysis_with_mechanistic_interpretability.git
   cd cross_modal_analysis_with_mechanistic_interpretability
   python -m venv .venv
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # On Windows PowerShell
   
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
## 📁 Repository Structure
    
    ├── Cross-modal Analysis with Mechanistic Interpretability.py
    ├── ViT-Prisma/                      # Submodule for ViT-based interpretability
    ├── data/                            # Placeholder for input datasets
    ├── notebooks/                       # Jupyter notebooks for exploratory analysis
    ├── README.md
    └── requirements.txt

## 🧪 Current Experiments
- Layer-wise probing of ViT attention heads across modalities
- Alignment metrics between image and text embeddings
- Mechanistic tracing of feature propagation in multimodal fusion
## 🤝 Contributing
This project is in active development. Contributions are welcome, especially around:
- Mechanistic interpretability techniques
- Cross-modal evaluation datasets
- Visualisation tools for internal model states

Please open an issue or pull request with your ideas.
## 🧠 Design
I am designing this platform based on user feedback and research. Please feel free to email me at laranguyen811@gmail.com. Alternatively, please create an issue for the repository. 

## 📜 License
This repository is licensed under the MIT License. See LICENSE for details.

## 🌱 Ethical Framing
This work is guided by principles of public-good impact, transparency, sustainability and multispecies justice. It aims to advance interpretability in AI systems while remaining accountable to broader ecological and social contexts.

## 🙏 Acknowledgments
Inspired by ongoing work in AI safety, interpretability, and collaborative inquiry. Built with gratitude for open-source communities and systems thinkers everywhere.
This is an ongoing project — iterative, imperfect, and evolving. If you notice issues, have suggestions, or want to share reflections, please feel free to leave feedback via [issues](https://github.com/Laranguyen811/cross_modal_analysis_with_mechanistic_interpretability/issues) or pull requests. Your insights are welcome and appreciated.

This project uses the [Prisma Framework](https://github.com/Prisma-Multimodal/ViT-Prisma),  originally developed by Sonia Joseph and contributors,  licensed under the [MIT License](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/LICENSE). We gratefully acknowledge their work on mechanistic interpretability in multimodal models.

## 🤖 AI-Assisted Collaboration

This project has benefited from the use of AI assistants for programming, research, and linguistic enrichment.  
We gratefully acknowledge the following:

- **Microsoft Copilot**  
  Supported legal scaffolding, modular templates, and emotionally intelligent phrasing.  
  [https://copilot.microsoft.com](https://copilot.microsoft.com)

- **Gemini (Google DeepMind)**  
  Assisted with troubleshooting, code refactoring, and conceptual brainstorming.  
  [https://deepmind.google](https://deepmind.google)

- **Claude (Anthropic)**  
  Contributed to ethical reasoning scaffolds, interpretability, and collaborative drafting.  
  [https://www.anthropic.com](https://www.anthropic.com)

These tools were used as collaborative assistants in both programming and research contexts.  
The project author made all final decisions, edits and validations.  
Attribution reflects usage transparency and ethical engagement with generative technologies.


