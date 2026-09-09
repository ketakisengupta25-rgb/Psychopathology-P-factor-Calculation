# GOASSESS p-Factor Bifactor IRT Analysis

A reproducible psychometric pipeline for estimating a general psychopathology **p factor** and four specific dimensions from GOASSESS questionnaire responses using bifactor item response theory.

## Overview

The analysis performs questionnaire quality control, response recoding, confirmatory bifactor IRT estimation, EAP factor scoring, and export of participant-level factor scores and item-level factor loadings.

The bifactor structure contains:

- one **general factor (G)** loading on all modeled items;
- four orthogonal **specific factors (S1-S4)**;
- 112 modeled questionnaire items in total.

The specific-factor structure contains 37 items on S1 and 25 items each on S2, S3, and S4.

## Analysis pipeline

The script performs the following steps:

1. Loads GOASSESS questionnaire data.
2. Retains participants who pass all eight catch questions.
3. Removes catch-question columns.
4. Converts questionnaire responses to numeric values.
5. Recodes selected SIP items from seven response categories to two categories.
6. Removes participants with missing questionnaire responses.
7. Calculates response-quality statistics:
   - Long String Index
   - Mahalanobis distance
8. Excludes participants with a Long String Index greater than 20.
9. Flags multivariate Mahalanobis outliers at the 99th percentile.
10. Reverse-scores questionnaire items.
11. Renames questionnaire columns to their psychometric item identifiers.
12. Selects and orders the 112 p-factor items.
13. Fits a confirmatory bifactor IRT model.
14. Calculates EAP estimates for the general and four specific factors.
15. Exports factor scores, factor loadings, model datasets, quality-control statistics, and the fitted model.

## Model estimation

The confirmatory bifactor model is estimated using **quasi-Monte Carlo expectation-maximization (QMCEM)** with Sobol sampling. This avoids the rapid growth in computational cost associated with multidimensional Cartesian quadrature for a model containing five latent dimensions.

Participant factor scores are calculated as expected-a-posteriori (**EAP**) estimates using Sobol quasi-Monte Carlo integration under a standard multivariate-normal latent prior.

## Repository structure

```text
goassess-p-factor-irt/
│
├── goassess_p_factor_irt.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── raw/
│
└── outputs/
```

## Requirements

- Python 3.11 or later
- NumPy
- pandas
- SciPy
- openpyxl
- mirt

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Input data

Raw participant data are not included in this repository.

Place the GOASSESS dataset in:

```text
data/raw/goassess.xlsx
```

The questionnaire is expected to contain a participant identifier in the first column and the original question columns (`Q1`, `Q2`, etc.) used by the analysis.

If the input filename is different, edit:

```python
DATA_FILE = Path("data/raw/goassess.xlsx")
```

near the top of `goassess_p_factor_irt.py`.

## Running the analysis

From the repository root, run:

```bash
python goassess_p_factor_irt.py
```

Model estimation can be computationally intensive because the bifactor model contains five latent dimensions and 112 items.

## Outputs

The script creates the following files in `outputs/`:

```text
Goassess_factor_scores_final_reversed.xlsx
Goassess_factor_loadings_final_reversed.xlsx
Model_data_reversed.xlsx
Model_data_original.xlsx
Response_quality_statistics.xlsx
goassess_p_bifactor_model_reversed.pkl
```

### Factor scores

`Goassess_factor_scores_final_reversed.xlsx` contains:

- `ParticipantID`
- `G` — general psychopathology p-factor score
- `S1`
- `S2`
- `S3`
- `S4`

### Factor loadings

`Goassess_factor_loadings_final_reversed.xlsx` contains each questionnaire item and its loading on:

- `G`
- `S1`
- `S2`
- `S3`
- `S4`

### Response-quality statistics

`Response_quality_statistics.xlsx` contains participant-level Long String Index and Mahalanobis-distance information.

By default, Mahalanobis outliers are **flagged but not excluded** from model estimation. This behavior can be changed in the configuration section:

```python
EXCLUDE_MAHALANOBIS_OUTLIERS = True
```

## Data privacy

Raw questionnaire data, model outputs, Excel files, and serialized fitted models are excluded from version control through `.gitignore`. Participant-level datasets should not be committed to a public repository.

## Author

Ketaki Sengupta
