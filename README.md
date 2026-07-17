# Job Posting NLP Pipeline

This project uses job descriptions for two tasks:

* **classifying postings** into seven role categories;
* **retrieving relevant postings** from natural-language search queries.

The notebook covers the data audit, preprocessing, model comparison, final classification evaluation and semantic-search implementation.

## Results

Models were compared using grouped five-fold cross-validation on the development set. The selected model was then evaluated once on the held-out test set.

| Evaluation        | Approach                         |          Macro F1 |       Weighted F1 |          Accuracy |
| ----------------- | -------------------------------- | ----------------: | ----------------: | ----------------: |
| Development CV    | Majority baseline                |     0.085 ± 0.000 |     0.254 ± 0.001 |     0.426 ± 0.001 |
| Development CV    | Keyword heuristic                |     0.517 ± 0.014 |     0.662 ± 0.012 |     0.590 ± 0.013 |
| Development CV    | TF-IDF + logistic regression     |     0.760 ± 0.020 | **0.858 ± 0.008** | **0.858 ± 0.008** |
| Development CV    | MiniLM + logistic regression     |     0.673 ± 0.008 |     0.785 ± 0.008 |     0.777 ± 0.008 |
| Development CV    | MPNet + logistic regression      |     0.736 ± 0.013 |     0.835 ± 0.006 |     0.829 ± 0.007 |
| Development CV    | MPNet + small neural network     | **0.762 ± 0.019** |     0.849 ± 0.008 |     0.845 ± 0.008 |
| **Held-out test** | **TF-IDF + logistic regression** |         **0.768** |         **0.873** |         **0.873** |

The final classifier uses TF-IDF word unigrams with `min_df=2` and balanced logistic regression (`C=1.0`, `solver="lbfgs"`).

Although the MPNet neural network had a slightly higher development macro F1, the difference was very small. I selected TF-IDF because it achieved better weighted F1 and accuracy, used the full descriptions and was much cheaper to run on CPU.

### Semantic search

Semantic search uses `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` over 12,941 chunks from 3,622 unique cleaned descriptions.

A small evaluation using four queries and 20 manual relevance judgments produced:

* mean Precision@5: **0.950**
* mean nDCG@5: **0.990**

Given the small evaluation set, these results should be treated as a demonstration rather than a broad benchmark.

## Repository structure

```text
.
├── job_posting_nlp_take_home_task.ipynb  # Main analysis and results
├── data/                                 # Supplied job-posting data
├── src/job_posting_nlp/
│   ├── preprocessing.py                  # Text preprocessing
│   ├── baselines.py                      # Keyword baseline
│   ├── classification.py                 # TF-IDF classification
│   ├── embeddings.py                     # Dense embedding generation
│   ├── neural_network.py                 # Small embedding classifier
│   ├── evaluation.py                     # Splitting and evaluation utilities
│   └── retrieval.py                      # Chunking, caching and semantic search
├── tests/                                # Unit tests
├── pyproject.toml                        # Project configuration
└── uv.lock                               # Locked dependencies
```

## Setup

The project requires Python 3.12 or newer.

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```bash
uv sync
```

Open the notebook with:

```bash
uv run jupyter lab
```

Run the tests with:

```bash
uv run pytest
```

No API keys are required.

## Running the analysis

The main notebook is:

```text
job_posting_nlp_take_home_task.ipynb
```

It should be run from the repository root, with the supplied CSV files kept under `data/`.

Sentence Transformer models are downloaded from Hugging Face when they are not already available locally. Dense embedding generation can take several minutes on CPU. In the saved run, MPNet encoding took approximately 1,107 seconds and retrieval encoding took approximately 386 seconds.

Retrieval embeddings are cached under `.cache/embeddings/`. This directory is ignored by Git, and the embeddings are regenerated when the cache is missing.

## Limitations

The main limitations are the title-derived labels, class imbalance, and the limited retrieval evaluation. These are discussed in more detail in the notebook.
