# Multilingual WEAT/SEAT Bias Evaluation

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)

Multilingual Word Embedding Association Test (WEAT) and Sentence Embedding Association Test (SEAT) evaluation framework for measuring gender, racial, ethnic, regional, and sentiment bias in embedding models across five languages.

## Overview

This repository contains the complete code, data, and results for the paper:

**"Multilingual WEAT Analysis of Proprietary and Open-Weight Embedding Models Across Five Languages"**

Submitted to *Journal of Information and Data Management (JIDM)*.

## Key Features

- **9 embedding models**: 4 proprietary (OpenAI text-embedding-3-small/large, Google Gemini Embedding 001/2) + 5 open-weight (Qwen3-Embedding-8B/4B, BGE-M3, Multilingual-E5-Large, Mistral-Embed)
- **5 languages**: Brazilian Portuguese, English, Spanish, French, German
- **432 experiments** (9 models × 48 bias dimensions)
- **WEAT + SEAT** methodology with permutation testing (n=5,000) and percentile bootstrap CIs (n=10,000)
- **Holm-Bonferroni correction** per (model, language) family
- **Culturally adapted word lists** for ES, FR, DE (4 minority groups each, based on census/migration data)
- **IBGE racial categories** for PT-BR (sentiment + status, word-level + SEAT sentence-level)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"

# Run all experiments
python src/run_experiments.py

# Generate analysis and figures
python src/analysis.py
```

## Repository Structure

```
bias_embedding/
├── data/
│   └── weat_lists/           # Word lists for all 5 languages
├── results/
│   ├── *.json                # Per-model-per-language results
│   ├── results.csv           # All 432 experiments
│   └── summary.json          # Aggregated statistics
├── src/
│   ├── weat.py               # WEAT/SEAT implementation
│   ├── analysis.py           # Statistical analysis + figures
│   ├── embeddings.py         # OpenRouter API wrapper
│   └── run_experiments.py    # Main experiment runner
├── data/weat_lists/          # Word lists (PT-BR, EN, ES, FR, DE)
├── paper/
│   └── jidm/                 # JIDM 2024 submission
│       ├── main.tex
│       ├── main.pdf
│       ├── refs.bib
│       └── cover_letter.pdf
└── paper/figures/            # Generated figures
```

## Key Results

| Finding | Details |
|---------|---------|
| **Sentiment bias** | Most consistent across all 5 languages (flowers/insects, instruments/weapons) |
| **Gender bias** | Non-significant in 4/5 languages after correction; only Qwen3-8B in DE significant |
| **Racial/ethnic bias** | Culture-specific: Maghrebi in FR (5/9), Turkish in DE (4/9), Moroccan/Sub-Saharan in ES (3/9) |
| **Proprietary vs Open-weight** | No systematic difference in bias magnitude |
| **Model size vs bias** | Positive correlation (Spearman ρ=1.00 for params, ρ=0.70 for dims) — scaling alone doesn't reduce bias |

## Data Availability

All code, word lists, raw results (CSV + JSON), and analysis scripts are available in this repository under CC BY 4.0.

- **Results**: `results/results.csv` (432 rows)
- **Word lists**: `data/weat_lists/weat_{ptbr,en,es,fr,de}.json`
- **Code**: `src/weat.py`, `src/analysis.py`

## Citation

```bibtex
@article{sansao2026multilingual,
  title={Multilingual WEAT Analysis of Proprietary and Open-Weight Embedding Models Across Five Languages},
  author={Sansão, João Pedro Hallack},
  journal={Journal of Information and Data Management},
  year={2026},
  note={Submitted}
}
```

## License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Code is provided for research purposes. See LICENSE for details.

## Paper

The paper is under review at *Journal of Information and Data Management (JIDM)*. Preprint available in `paper/jidm/main.pdf`.
