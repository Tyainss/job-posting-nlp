# Data

The project is based on the public [LinkedIn Job Postings dataset on Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings).

The saved analysis uses a pre-filtered tech/data subset with the following derived files:

| File | Rows | Purpose |
| --- | ---: | --- |
| `sample_postings.csv` | 4,072 | Main modelling table containing job descriptions and `role_category` |
| `sample_companies.csv` | 1,979 | Company metadata |
| `sample_skills.csv` | 6,800 | Job-to-skill mappings |
| `sample_skill_lookup.csv` | 35 | Skill-code lookup |
| `sample_industries.csv` | 422 | Industry lookup |

`role_category` is a weak target derived from job-title keyword rules. It is useful for evaluating the classification workflow, but it should not be interpreted as manually reviewed ground truth.

## Reproducibility

The derived CSV files are not committed to the repository and are ignored by Git.

The repository contains the modelling, evaluation and retrieval workflow once these files are available. The exact filtering and label-generation logic that produced this subset is not included, so regenerating the same derived CSVs from the raw Kaggle dataset is outside the current repository scope.
