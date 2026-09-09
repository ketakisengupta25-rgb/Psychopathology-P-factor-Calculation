"""
GOASSESS p-Factor Bifactor IRT Pipeline

This script prepares GOASSESS questionnaire data, performs response-quality
screening, fits a confirmatory bifactor item response theory model, estimates
participant factor scores, and exports factor loadings and model datasets.

Model structure

- General factor: p factor loading on all 112 modeled items
- Specific factor 1: first 37 items
- Specific factor 2: next 25 items
- Specific factor 3: next 25 items
- Specific factor 4: final 25 items

The model is estimated with quasi-Monte Carlo EM (QMCEM), and participant
factor scores are calculated as expected-a-posteriori (EAP) estimates using
Sobol quasi-Monte Carlo integration.

Author: Ketaki Sengupta
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import pinvh
from scipy.stats import chi2, norm, qmc

from mirt import BifactorModel
from mirt.estimation import QMCEMEstimator



# CONFIGURATION


DATA_FILE = Path("data/raw/goassess.xlsx")
OUTPUT_DIR = Path("outputs")

LONG_STRING_MAX = 20
MAHALANOBIS_ALPHA = 0.99

# The original analysis identified Mahalanobis outliers but did not remove them.
# Set this to True if they should also be excluded from model estimation.
EXCLUDE_MAHALANOBIS_OUTLIERS = False

# QMC model-estimation settings.
QMC_SAMPLES = 2048
MAX_ITER = 5000
TOLERANCE = 1e-4
RANDOM_SEED = 42

# Number of Sobol draws used for EAP factor scoring.
EAP_QMC_SAMPLES = 8192

CATCH_QUESTIONS = {
    "Q16": 2,
    "Q31": 2,
    "Q47": 2,
    "Q65": 1,
    "Q81": 2,
    "Q104": 6,
    "Q114": 2,
    "Q117": 2,
}

SIP_ITEMS_TO_BINARY = [
    "Q95", "Q96", "Q97", "Q98", "Q99", "Q100",
    "Q101", "Q102", "Q103", "Q105", "Q106", "Q107",
]

MAHALANOBIS_EXCLUDE = ["Q22", "Q27", "Q28", "Q29"]

NEW_COLUMN_NAMES = [
    "Participant",
    "ADD011", "ADD012", "ADD013", "ADD014", "ADD015",
    "ADD016", "ADD020", "ADD021", "ADD022",
    "AGR001", "AGR002", "AGR003", "AGR004", "AGR005",
    "AGR006", "AGR007", "AGR008", "CDD001", "CDD002",
    "CDD003", "CDD004", "CDD005", "CDD006", "CDD007",
    "CDD008", "CDD009", "CDD010", "CDD011", "DEP001",
    "DEP002", "DEP004", "DEP006", "GAD001", "GAD002",
    "MAN001", "MAN002", "MAN003", "MAN004", "MAN005",
    "MAN006", "MAN007", "OCD001", "OCD002", "OCD003",
    "OCD004", "OCD005", "OCD006", "OCD007", "OCD008",
    "OCD011", "OCD012", "OCD013", "OCD014", "OCD015",
    "OCD016", "OCD017", "OCD018", "OCD019", "ODD001",
    "ODD002", "ODD003", "ODD005", "ODD006", "PAN001",
    "PAN003", "PAN004", "PHB001", "PHB002", "PHB003",
    "PHB004", "PHB005", "PHB006", "PHB007", "PHB008",
    "PSY001", "PSY020", "PSY029", "PSY050", "PSY060",
    "PSY070", "PSY071", "SCR001", "SCR006", "SCR007",
    "SEP500", "SEP508", "SEP509", "SEP510", "SEP511",
    "SIP003", "SIP004", "SIP005", "SIP006", "SIP007",
    "SIP008", "SIP009", "SIP010", "SIP011", "SIP012",
    "SIP013", "SIP014", "SIP027", "SIP028", "SIP032",
    "SIP033", "SIP038", "SIP039", "SOC001", "SOC002",
    "SOC003", "SOC004", "SOC005",
]

P_FACTOR_ITEMS = [
    "DEP001", "OCD007", "GAD002", "DEP002", "OCD001", "SIP032",
    "PAN001", "OCD005", "OCD016", "PAN004", "GAD001", "PAN003",
    "DEP006", "OCD003", "SCR007", "MAN007", "OCD012", "DEP004",
    "OCD004", "OCD006", "OCD011", "OCD019", "OCD013", "OCD008",
    "SCR001", "SIP039", "OCD018", "MAN004", "SIP033", "OCD017",
    "MAN005", "OCD002", "OCD014", "SCR006", "OCD015", "SEP510",
    "SIP038",

    "ODD002", "CDD010", "CDD003", "CDD008", "CDD005", "CDD001",
    "CDD007", "ODD005", "CDD009", "ODD001", "ADD016", "CDD002",
    "ODD003", "ADD012", "CDD006", "ADD011", "ODD006", "CDD004",
    "ADD021", "ADD013", "ADD014", "ADD022", "ADD020", "ADD015",
    "CDD011",

    "SIP012", "SIP007", "SIP010", "SIP008", "SIP011", "SIP013",
    "SIP005", "SIP003", "SIP004", "SIP006", "SIP009", "PSY001",
    "PSY029", "SIP014", "PSY060", "PSY020", "PSY070", "MAN006",
    "PSY050", "MAN003", "MAN002", "MAN001", "PSY071", "SIP027",
    "SIP028",

    "AGR006", "AGR005", "SOC004", "AGR008", "SOC003", "AGR004",
    "AGR001", "SOC005", "SOC001", "PHB004", "AGR002", "AGR003",
    "PHB007", "SOC002", "PHB006", "PHB001", "AGR007", "PHB002",
    "SEP509", "PHB003", "PHB005", "SEP508", "PHB008", "SEP500",
    "SEP511",
]

# 37 items on S1, followed by 25 each on S2, S3, and S4.
SPECIFIC_FACTORS = np.array(
    [1] * 37
    + [2] * 25
    + [3] * 25
    + [4] * 25,
    dtype=int,
)



# DATA LOADING AND VALIDATION


def load_data(file_path: Path) -> pd.DataFrame:
    """Load GOASSESS data from an Excel or CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}\n"
            "Place the GOASSESS file in data/raw/ and update DATA_FILE if needed."
        )

    suffix = file_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    if suffix == ".csv":
        return pd.read_csv(file_path)

    raise ValueError("DATA_FILE must be an .xlsx, .xls, or .csv file.")


def validate_required_columns(data: pd.DataFrame) -> None:
    """Check that all questionnaire columns required by the pipeline exist."""
    required = set(CATCH_QUESTIONS) | set(SIP_ITEMS_TO_BINARY)

    missing = sorted(required - set(data.columns))

    if missing:
        raise ValueError(
            "The following required questionnaire columns are missing: "
            + ", ".join(missing)
        )



# RESPONSE-QUALITY SCREENING

def filter_attention_checks(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only participants who answered all catch questions correctly."""
    mask = pd.Series(True, index=data.index)

    for item, correct_answer in CATCH_QUESTIONS.items():
        mask &= pd.to_numeric(data[item], errors="coerce").eq(correct_answer)

    filtered = data.loc[mask].copy()

    print(
        f"Attention checks: retained {len(filtered)} of {len(data)} participants."
    )

    return filtered.drop(columns=list(CATCH_QUESTIONS))


def convert_item_columns_to_numeric(data: pd.DataFrame) -> pd.DataFrame:
    """Convert all questionnaire columns except participant ID to numeric."""
    data = data.copy()

    participant_col = data.columns[0]

    for column in data.columns:
        if column != participant_col:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def recode_sip_items(data: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse specified SIP items from seven response categories to two.

    Original scoring:
        1-2 -> 1
        3-7 -> 2
    """
    data = data.copy()

    mapping = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
    }

    for item in SIP_ITEMS_TO_BINARY:
        data[item] = data[item].map(mapping)

    return data


def calculate_long_string_index(responses: np.ndarray) -> int:
    """Return the longest run of identical consecutive responses."""
    if len(responses) == 0:
        return 0

    max_length = 1
    current_length = 1

    for index in range(1, len(responses)):
        if responses[index] == responses[index - 1]:
            current_length += 1
        else:
            max_length = max(max_length, current_length)
            current_length = 1

    return max(max_length, current_length)


def add_long_string_index(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the long-string index for every participant."""
    data = data.copy()

    response_matrix = data.iloc[:, 1:].to_numpy()

    data["Long_String_Index"] = [
        calculate_long_string_index(row)
        for row in response_matrix
    ]

    return data


def calculate_mahalanobis_qc(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate squared Mahalanobis distance for each participant.

    The chi-square cutoff is applied to squared Mahalanobis distance, which
    places the statistic and threshold on the same scale.
    """
    participant_col = data.columns[0]

    item_data = data.drop(columns=[participant_col], errors="ignore")
    item_data = item_data.drop(columns=MAHALANOBIS_EXCLUDE, errors="ignore")

    matrix = item_data.to_numpy(dtype=float)

    mean_vector = np.mean(matrix, axis=0)
    correlation_matrix = np.corrcoef(matrix, rowvar=False)

    # Pseudoinverse is more stable than a direct inverse if items are highly
    # correlated or the correlation matrix is close to singular.
    inverse_correlation = pinvh(correlation_matrix)

    centered = matrix - mean_vector

    squared_distances = np.einsum(
        "ij,jk,ik->i",
        centered,
        inverse_correlation,
        centered,
    )

    degrees_of_freedom = matrix.shape[1]
    threshold = chi2.ppf(MAHALANOBIS_ALPHA, df=degrees_of_freedom)

    return pd.DataFrame(
        {
            "Participant": data[participant_col].to_numpy(),
            "Mahalanobis_D2": squared_distances,
            "Mahalanobis_Threshold": threshold,
            "Mahalanobis_Outlier": squared_distances > threshold,
        }
    )



# SCORING AND MODEL PREPARATION

def reverse_score_items(data: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse binary item scoring using 3 - response.

    1 becomes 2, and 2 becomes 1.
    """
    data = data.copy()

    data.iloc[:, 1:] = 3 - data.iloc[:, 1:]

    return data


def rename_items(data: pd.DataFrame) -> pd.DataFrame:
    """Assign the psychometric item names used by the bifactor model."""
    if data.shape[1] != len(NEW_COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(NEW_COLUMN_NAMES)} columns after removing catch "
            f"questions, but found {data.shape[1]}. Check the raw questionnaire "
            "structure before fitting the model."
        )

    renamed = data.copy()
    renamed.columns = NEW_COLUMN_NAMES

    return renamed


def create_p_factor_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Select and order the 112 items used in the p-factor model."""
    missing = [item for item in P_FACTOR_ITEMS if item not in data.columns]

    if missing:
        raise ValueError(
            "Model items missing after renaming: " + ", ".join(missing)
        )

    return data[P_FACTOR_ITEMS].copy()


def to_binary_zero_one(p_data: pd.DataFrame) -> np.ndarray:
    """
    Convert reversed 1/2 responses to the 0/1 coding expected by dichotomous IRT.

    Reversed score 1 -> 0
    Reversed score 2 -> 1
    """
    unique_values = set(np.unique(p_data.to_numpy(dtype=int)))

    if not unique_values.issubset({1, 2}):
        raise ValueError(
            "The bifactor model expects every modeled item to contain only "
            f"1/2 responses after preprocessing. Found: {sorted(unique_values)}"
        )

    return p_data.to_numpy(dtype=int) - 1

# BIFACTOR IRT

def fit_bifactor_model(responses: np.ndarray):
    """Fit the four-specific-factor bifactor IRT model using QMCEM."""
    if responses.shape[1] != len(SPECIFIC_FACTORS):
        raise ValueError(
            "The number of modeled items does not match the factor structure."
        )

    model = BifactorModel(
        n_items=responses.shape[1],
        specific_factors=SPECIFIC_FACTORS,
        item_names=P_FACTOR_ITEMS,
    )

    estimator = QMCEMEstimator(
        n_samples=QMC_SAMPLES,
        max_iter=MAX_ITER,
        tol=TOLERANCE,
        verbose=True,
        seed=RANDOM_SEED,
        sequence="sobol",
    )

    print("\nFitting bifactor IRT model...")
    fit_result = estimator.fit(model, responses)

    return fit_result


def calculate_eap_qmc_scores(
    model: BifactorModel,
    responses: np.ndarray,
    n_samples: int = EAP_QMC_SAMPLES,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Estimate EAP factor scores using Sobol quasi-Monte Carlo integration.

    Samples are drawn from the standard multivariate-normal latent prior.
    Posterior weights are proportional to the response-pattern likelihood.
    """
    n_persons = responses.shape[0]
    n_factors = model.n_factors

    sampler = qmc.Sobol(
        d=n_factors,
        scramble=True,
        seed=seed,
    )

    exponent = int(np.ceil(np.log2(n_samples)))
    uniforms = sampler.random_base2(exponent)[:n_samples]

    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)

    theta = norm.ppf(np.clip(uniforms, lower, upper))

    probabilities = np.clip(
        model.probability(theta),
        1e-12,
        1 - 1e-12,
    )

    log_p = np.log(probabilities)
    log_q = np.log1p(-probabilities)

    scores = np.empty((n_persons, n_factors), dtype=float)

    for person_index, person_responses in enumerate(responses):
        log_likelihood = (
            person_responses[None, :] * log_p
            + (1 - person_responses[None, :]) * log_q
        ).sum(axis=1)

        # Stabilize exponentiation.
        log_likelihood -= log_likelihood.max()

        weights = np.exp(log_likelihood)
        weights /= weights.sum()

        scores[person_index] = weights @ theta

    return scores


def extract_factor_loadings(model: BifactorModel) -> pd.DataFrame:
    """Return item loadings for the general and four specific factors."""
    loading_matrix = model.get_loading_matrix()

    loadings = pd.DataFrame(
        loading_matrix,
        columns=["G", "S1", "S2", "S3", "S4"],
    )

    loadings.insert(0, "Item", P_FACTOR_ITEMS)

    return loadings


# EXPORT

def save_outputs(
    original_data: pd.DataFrame,
    reversed_data: pd.DataFrame,
    qc_data: pd.DataFrame,
    participant_ids: pd.Series,
    responses: np.ndarray,
    fit_result,
) -> None:
    """Save participant scores, loadings, datasets, QC statistics, and model."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = fit_result.model

    factor_scores_array = calculate_eap_qmc_scores(
        model,
        responses,
    )

    factor_scores = pd.DataFrame(
        factor_scores_array,
        columns=["G", "S1", "S2", "S3", "S4"],
    )

    factor_scores["ParticipantID"] = participant_ids.to_numpy()

    # Put participant ID first for easier downstream merging.
    factor_scores = factor_scores[
        ["ParticipantID", "G", "S1", "S2", "S3", "S4"]
    ]

    factor_loadings = extract_factor_loadings(model)

    factor_scores.to_excel(
        OUTPUT_DIR / "Goassess_factor_scores_final_reversed.xlsx",
        index=False,
    )

    factor_loadings.to_excel(
        OUTPUT_DIR / "Goassess_factor_loadings_final_reversed.xlsx",
        index=False,
    )

    reversed_data.to_excel(
        OUTPUT_DIR / "Model_data_reversed.xlsx",
        index=False,
    )

    original_data.to_excel(
        OUTPUT_DIR / "Model_data_original.xlsx",
        index=False,
    )

    qc_data.to_excel(
        OUTPUT_DIR / "Response_quality_statistics.xlsx",
        index=False,
    )

    with open(
        OUTPUT_DIR / "goassess_p_bifactor_model_reversed.pkl",
        "wb",
    ) as model_file:
        pickle.dump(fit_result, model_file)

    print("\nSaved:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file() and path.name != ".gitkeep":
            print(f"  - {path}")



# MAIN PIPELINE

def main() -> None:
    """Run the complete GOASSESS p-factor IRT analysis."""
    print("Loading GOASSESS data...")
    raw_data = load_data(DATA_FILE)

    validate_required_columns(raw_data)

    
    # Attention checks
    
    goassess = filter_attention_checks(raw_data)

    
    # Numeric conversion and SIP recoding
    
    goassess = convert_item_columns_to_numeric(goassess)
    goassess = recode_sip_items(goassess)

    before_missing = len(goassess)
    goassess = goassess.dropna().reset_index(drop=True)

    print(
        f"Missing data: removed {before_missing - len(goassess)} participants."
    )

    # Store the scored-but-not-reversed version for export.
    goassess_original = goassess.copy()

    
    # Long-string quality control
    
    goassess_lsi = add_long_string_index(goassess)

    mean_lsi = goassess_lsi["Long_String_Index"].mean()

    print(f"Mean Long String Index: {mean_lsi:.2f}")

    
    # Mahalanobis quality control
    
    mahalanobis_qc = calculate_mahalanobis_qc(goassess)

    qc_data = mahalanobis_qc.merge(
        goassess_lsi[
            [goassess_lsi.columns[0], "Long_String_Index"]
        ].rename(columns={goassess_lsi.columns[0]: "Participant"}),
        on="Participant",
        how="left",
    )

    qc_data["Long_String_Excluded"] = (
        qc_data["Long_String_Index"] > LONG_STRING_MAX
    )

    
    # Apply participant exclusions
    
    keep_mask = goassess_lsi["Long_String_Index"] <= LONG_STRING_MAX

    if EXCLUDE_MAHALANOBIS_OUTLIERS:
        keep_mask &= ~mahalanobis_qc["Mahalanobis_Outlier"].to_numpy()

    goassess_clean = (
        goassess.loc[keep_mask.to_numpy()]
        .reset_index(drop=True)
    )

    print(
        f"Long-string criterion (<= {LONG_STRING_MAX}): "
        f"retained {len(goassess_clean)} participants."
    )

    if EXCLUDE_MAHALANOBIS_OUTLIERS:
        print(
            "Mahalanobis outliers were excluded from model estimation."
        )
    else:
        print(
            "Mahalanobis outliers were flagged but not excluded "
            "(EXCLUDE_MAHALANOBIS_OUTLIERS=False)."
        )

    # Align exported original data with the model sample.
    goassess_original_clean = goassess_clean.copy()

    
    # Reverse scoring and item renaming
    
    goassess_reversed = reverse_score_items(goassess_clean)

    goassess_original_named = rename_items(goassess_original_clean)
    goassess_reversed_named = rename_items(goassess_reversed)

    
    # Create p-factor response matrix
    
    p_data = create_p_factor_matrix(goassess_reversed_named)

    responses = to_binary_zero_one(p_data)

    print(
        f"\nModel sample: {responses.shape[0]} participants x "
        f"{responses.shape[1]} items."
    )

    
    # Fit bifactor model
    
    fit_result = fit_bifactor_model(responses)

    print("\nModel fitting complete.")
    print(fit_result.summary())

    
    # Export results
    
    save_outputs(
        original_data=goassess_original_named,
        reversed_data=goassess_reversed_named,
        qc_data=qc_data,
        participant_ids=goassess_reversed_named["Participant"],
        responses=responses,
        fit_result=fit_result,
    )


if __name__ == "__main__":
    main()
