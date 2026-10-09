
"""
Sepsis AI — Frozen Longitudinal Feature Engineering Service

Purpose
-------
Reproduce the approved Phase 03 leakage-safe feature-engineering logic
for the frozen B03 Sepsis AI research prototype.

This module:

1. accepts longitudinal hourly patient observations;
2. preserves the Phase 03 information boundary;
3. creates the approved temporal representations;
4. returns predictors in the exact frozen registry order.

It does NOT:

- fit or refit B03;
- fit preprocessing parameters;
- recalibrate probabilities;
- modify OP07;
- access the held-out Test set;
- create causal or diagnostic claims;
- authorize clinical deployment.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================================================
# FROZEN SOURCE VARIABLES
# ============================================================================================

TIME_VARYING_FEATURES = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
]


# Phase 03 approved Tier 1: >=50% observed.
TIER1_FEATURES = [
    "HR",
    "MAP",
    "O2Sat",
    "SBP",
    "Resp",
    "DBP",
]


# Phase 03 approved Tier 2: >=10% and <50% observed.
TIER2_FEATURES = [
    "Temp",
    "Glucose",
]


STATIC_CONTEXT_FEATURES = [
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
]


HISTORY_FEATURES = [
    "PriorObservedHours",
    "ObservationNumber",
    "HasPriorHistory",
    "Has6PriorHours",
    "Has12PriorHours",
    "Has24PriorHours",
]


REQUIRED_LONGITUDINAL_COLUMNS = (
    ["PatientID", "ICULOS"]
    + TIME_VARYING_FEATURES
    + [
        "Age",
        "Gender",
        "Unit1",
        "Unit2",
        "HospAdmTime",
    ]
)


# ============================================================================================
# VALIDATION
# ============================================================================================

def validate_longitudinal_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate longitudinal prototype input without fitting or imputing anything.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Longitudinal patient input is empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_LONGITUDINAL_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required longitudinal columns are missing: "
            + ", ".join(missing_columns)
        )

    work = df.copy()

    if work["PatientID"].isna().any():
        raise ValueError(
            "PatientID contains missing values."
        )

    if work["ICULOS"].isna().any():
        raise ValueError(
            "ICULOS contains missing values."
        )

    duplicated_time = (
        work
        .duplicated(
            subset=["PatientID", "ICULOS"]
        )
        .any()
    )

    if duplicated_time:
        raise ValueError(
            "Duplicate PatientID + ICULOS observations detected."
        )

    work = (
        work
        .sort_values(
            ["PatientID", "ICULOS"]
        )
        .reset_index(drop=True)
    )

    # Exact hourly increments are required within each patient.
    hour_differences = (
        work
        .groupby(
            "PatientID",
            sort=False
        )["ICULOS"]
        .diff()
        .dropna()
    )

    if not hour_differences.eq(1).all():
        raise ValueError(
            "ICULOS must increase in exact one-hour increments "
            "within each patient trajectory."
        )

    return work


# ============================================================================================
# OBSERVATION-HISTORY FEATURES
# ============================================================================================

def add_observation_history(
    df: pd.DataFrame
) -> pd.DataFrame:

    work = df.copy()

    work["PriorObservedHours"] = (
        work
        .groupby(
            "PatientID",
            sort=False
        )
        .cumcount()
    )

    work["ObservationNumber"] = (
        work["PriorObservedHours"] + 1
    )

    work["HasPriorHistory"] = (
        work["PriorObservedHours"]
        .ge(1)
        .astype("int8")
    )

    work["Has6PriorHours"] = (
        work["PriorObservedHours"]
        .ge(6)
        .astype("int8")
    )

    work["Has12PriorHours"] = (
        work["PriorObservedHours"]
        .ge(12)
        .astype("int8")
    )

    work["Has24PriorHours"] = (
        work["PriorObservedHours"]
        .ge(24)
        .astype("int8")
    )

    return work


# ============================================================================================
# UNIVERSAL TEMPORAL REPRESENTATION
# ============================================================================================

def add_universal_temporal_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    work = df.copy()

    for feature in TIME_VARYING_FEATURES:

        observed = (
            work[feature]
            .notna()
        )

        # Current missingness.
        work[
            f"{feature}__missing"
        ] = (
            (~observed)
            .astype("int8")
        )

        # Last observed value at or before t.
        work[
            f"{feature}__last"
        ] = (
            work
            .groupby(
                "PatientID",
                sort=False
            )[feature]
            .ffill()
        )

        # Cumulative measurement count through t.
        work[
            f"{feature}__count"
        ] = (
            observed
            .groupby(
                work["PatientID"],
                sort=False
            )
            .cumsum()
            .astype("int32")
        )

        # Time since most recent measurement.
        last_measurement_time = (
            work["ICULOS"]
            .where(observed)
            .groupby(
                work["PatientID"],
                sort=False
            )
            .ffill()
        )

        work[
            f"{feature}__time_since"
        ] = (
            work["ICULOS"]
            - last_measurement_time
        )

    return work


# ============================================================================================
# PREVIOUS-OBSERVED-VALUE DELTA
# ============================================================================================

def add_previous_observed_delta(
    df: pd.DataFrame,
    feature: str
) -> None:

    last_col = (
        f"{feature}__last"
    )

    count_col = (
        f"{feature}__count"
    )

    delta_col = (
        f"{feature}__delta"
    )

    previous_observed = (
        df[last_col]
        .groupby(
            df["PatientID"],
            sort=False
        )
        .shift(1)
    )

    df[delta_col] = (
        df[last_col]
        - previous_observed
    )

    # Meaningful only after at least two actual measurements.
    df.loc[
        df[count_col] < 2,
        delta_col
    ] = np.nan

    # No new measurement at t -> delta remains missing.
    df.loc[
        df[feature].isna(),
        delta_col
    ] = np.nan


# ============================================================================================
# ICU-CLOCK ROLLING FEATURE
# ============================================================================================

def add_time_based_rolling_feature(
    df: pd.DataFrame,
    feature: str,
    window_hours: int,
    statistic: str
) -> None:

    output_col = (
        f"{feature}__roll"
        f"{window_hours}h_{statistic}"
    )

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64"
    )

    for _, idx in (
        df
        .groupby(
            "PatientID",
            sort=False
        )
        .groups
        .items()
    ):

        idx = np.asarray(idx)

        patient_hours = (
            df.loc[
                idx,
                "ICULOS"
            ]
            .to_numpy(
                dtype=float
            )
        )

        patient_values = (
            df.loc[
                idx,
                feature
            ]
            .to_numpy(
                dtype=float
            )
        )

        patient_series = pd.Series(
            patient_values,
            index=pd.to_timedelta(
                patient_hours,
                unit="h"
            )
        )

        rolling_object = (
            patient_series
            .rolling(
                f"{window_hours}h",
                closed="both",
                min_periods=1
            )
        )

        if statistic == "mean":
            rolled = (
                rolling_object.mean()
            )

        elif statistic == "min":
            rolled = (
                rolling_object.min()
            )

        elif statistic == "max":
            rolled = (
                rolling_object.max()
            )

        elif statistic == "std":
            rolled = (
                rolling_object.std(
                    ddof=1
                )
            )

        else:
            raise ValueError(
                "Unsupported rolling statistic: "
                f"{statistic}"
            )

        result.loc[idx] = (
            rolled.to_numpy()
        )

    df[output_col] = result


# ============================================================================================
# TIER-SPECIFIC TEMPORAL DYNAMICS
# ============================================================================================

def add_temporal_dynamics(
    df: pd.DataFrame
) -> pd.DataFrame:

    work = df.copy()

    for feature in TIER1_FEATURES:

        add_previous_observed_delta(
            work,
            feature
        )

        for window, statistic in [
            (6, "mean"),
            (6, "min"),
            (6, "max"),
            (6, "std"),
            (12, "mean"),
            (24, "mean"),
        ]:

            add_time_based_rolling_feature(
                work,
                feature,
                window,
                statistic
            )

    for feature in TIER2_FEATURES:

        add_previous_observed_delta(
            work,
            feature
        )

        for window, statistic in [
            (6, "mean"),
            (12, "mean"),
        ]:

            add_time_based_rolling_feature(
                work,
                feature,
                window,
                statistic
            )

    return work


# ============================================================================================
# COMPLETE FROZEN FEATURE CONSTRUCTION
# ============================================================================================

def engineer_frozen_features(
    longitudinal_df: pd.DataFrame,
    approved_feature_registry_path
):
    """
    Generate the complete frozen 230-predictor representation.

    Returns
    -------
    engineered_history:
        Longitudinal dataframe containing source and engineered fields.

    predictor_matrix:
        Exact 230 predictors in authoritative frozen registry order.
    """

    work = validate_longitudinal_input(
        longitudinal_df
    )

    work = add_observation_history(
        work
    )

    work = add_universal_temporal_features(
        work
    )

    work = add_temporal_dynamics(
        work
    )

    registry = pd.read_csv(
        approved_feature_registry_path
    )

    approved_predictors = (
        registry["FeatureName"]
        .tolist()
    )

    if len(approved_predictors) != 230:
        raise RuntimeError(
            "Frozen feature registry does not contain 230 predictors."
        )

    if len(set(approved_predictors)) != 230:
        raise RuntimeError(
            "Frozen feature registry contains duplicate predictor names."
        )

    missing_engineered_predictors = [
        feature
        for feature in approved_predictors
        if feature not in work.columns
    ]

    if missing_engineered_predictors:
        raise RuntimeError(
            "Frozen feature construction is incomplete. "
            "Missing predictors: "
            + ", ".join(
                missing_engineered_predictors
            )
        )

    predictor_matrix = (
        work[
            approved_predictors
        ]
        .copy()
    )

    if predictor_matrix.columns.tolist() != approved_predictors:
        raise RuntimeError(
            "Frozen predictor order reconciliation failed."
        )

    return work, predictor_matrix


# ============================================================================================
# LATEST-HOUR PROTOTYPE VIEW
# ============================================================================================

def latest_prediction_rows(
    engineered_history: pd.DataFrame,
    predictor_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Return only the latest observation for each patient.

    This does not perform model inference.
    """

    latest_indices = (
        engineered_history
        .groupby(
            "PatientID",
            sort=False
        )["ICULOS"]
        .idxmax()
    )

    latest_matrix = (
        predictor_matrix
        .loc[latest_indices]
        .copy()
        .reset_index(drop=True)
    )

    return latest_matrix
