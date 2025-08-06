
# Cross-modal Analysis with Mechanistic Interpretability

This repository explores cross-modal representations through the lens of mechanistic interpretability, with a focus on multimodal models. It is designed for iterative experimentation, diagnostic clarity, and reproducible workflows.

## 🔍 Project Overview

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
- Cross-modal benchmarking datasets
- Visualization tools for internal model states
Please open an issue or pull request with your ideas.

## 📜 License
This repository is licensed under the MIT License. See LICENSE for details.

## 🌱 Ethical Framing
This work is guided by principles of public-good impact, transparency, sustainability and multispecies justice. It aims to advance interpretability in AI systems while remaining accountable to broader ecological and social contexts.

## 🙏 Acknowledgments
Inspired by ongoing work in AI safety, interpretability, and collaborative inquiry. Built with gratitude for open-source communities and systems thinkers everywhere.
This is an ongoing project — iterative, imperfect, and evolving. If you notice issues, have suggestions, or want to share reflections, please feel free to leave feedback via [issues](https://github.com/Laranguyen811/cross_modal_analysis_with_mechanistic_interpretability/issues) or pull requests. Your insights are welcome and appreciated.

