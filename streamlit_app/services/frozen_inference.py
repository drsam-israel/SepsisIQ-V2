"""
Sepsis AI Early-Warning Prototype
Frozen B03 Inference Service

Developer: Dr. Samuel Israel

RESEARCH PROTOTYPE — NOT AUTHORIZED FOR CLINICAL DEPLOYMENT.

This module loads and executes the frozen B03 estimator.
It does not train, refit, recalibrate, or modify the operating threshold.
"""

from pathlib import Path
import hashlib
import pickle

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase04_modeling"
    / "B03_frozen_fitted_estimator.pkl"
)

FEATURE_REGISTRY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase03_features"
    / "approved_feature_registry.csv"
)

EXPECTED_MODEL_SHA256 = "d9dd6d01100bf82a0ba2ede5914b9294fe9bb87b4980c0df1584bf5edfc40a69"

MODEL_ID = "B03"
PREDICTOR_COUNT = 230

CALIBRATION_ALPHA = -0.868675893491
CALIBRATION_BETA = 0.858595710231
CALIBRATION_EPSILON = 1e-8

OPERATING_POINT = "OP07"
OPERATING_THRESHOLD = 0.004

RESEARCH_WARNING = (
    "RESEARCH PROTOTYPE — NOT AUTHORIZED FOR CLINICAL DEPLOYMENT"
)


def _sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


if not MODEL_PATH.exists():
    raise RuntimeError(
        "Frozen B03 estimator is unavailable."
    )


observed_sha = _sha256_file(
    MODEL_PATH
)

if observed_sha != EXPECTED_MODEL_SHA256:
    raise RuntimeError(
        "Frozen B03 estimator SHA mismatch. Inference blocked."
    )


feature_registry = pd.read_csv(
    FEATURE_REGISTRY_PATH
)


_candidate_columns = [
    "Feature",
    "FeatureName",
    "feature",
    "feature_name",
    "Predictor",
    "PredictorName",
    "predictor",
    "predictor_name"
]


_feature_name_column = next(
    (
        column
        for column in _candidate_columns
        if column in feature_registry.columns
    ),
    None
)


if _feature_name_column is None:
    raise RuntimeError(
        "Predictor-name column not found in frozen feature registry."
    )


APPROVED_FEATURES = (
    feature_registry[
        _feature_name_column
    ]
    .astype(str)
    .tolist()
)


if len(APPROVED_FEATURES) != PREDICTOR_COUNT:
    raise RuntimeError(
        "Frozen predictor count mismatch."
    )


if len(set(APPROVED_FEATURES)) != PREDICTOR_COUNT:
    raise RuntimeError(
        "Duplicate frozen predictor names detected."
    )


with open(
    MODEL_PATH,
    "rb"
) as handle:

    FROZEN_MODEL = pickle.load(
        handle
    )


def validate_payload(payload):

    if isinstance(payload, pd.Series):
        payload = payload.to_dict()

    if not isinstance(payload, dict):
        raise TypeError(
            "Inference payload must be a dictionary or pandas Series."
        )

    missing = [
        feature
        for feature in APPROVED_FEATURES
        if feature not in payload
    ]

    unexpected = [
        feature
        for feature in payload
        if feature not in APPROVED_FEATURES
    ]

    if missing:
        raise ValueError(
            "Missing frozen predictors: "
            + ", ".join(missing[:10])
        )

    if unexpected:
        raise ValueError(
            "Unexpected predictors: "
            + ", ".join(unexpected[:10])
        )

    ordered = {}

    for feature in APPROVED_FEATURES:

        value = payload[feature]

        if value is None:
            value = np.nan

        if not (
            pd.isna(value)
            or isinstance(
                value,
                (int, float, np.integer, np.floating)
            )
        ):
            raise TypeError(
                f"Predictor '{feature}' is not numeric or missing."
            )

        ordered[feature] = value

    frame = pd.DataFrame(
        [ordered],
        columns=APPROVED_FEATURES
    )

    if frame.shape != (1, PREDICTOR_COUNT):
        raise RuntimeError(
            "Frozen inference matrix shape mismatch."
        )

    return frame


def calibrate_probability(raw_probability):

    raw_probability = float(
        raw_probability
    )

    if not np.isfinite(raw_probability):
        raise ValueError(
            "Raw probability is not finite."
        )

    if not 0.0 <= raw_probability <= 1.0:
        raise ValueError(
            "Raw probability is outside [0, 1]."
        )

    clipped = np.clip(
        raw_probability,
        CALIBRATION_EPSILON,
        1.0 - CALIBRATION_EPSILON
    )

    raw_logit = np.log(
        clipped / (1.0 - clipped)
    )

    calibrated_logit = (
        CALIBRATION_ALPHA
        +
        CALIBRATION_BETA
        *
        raw_logit
    )

    return float(
        1.0
        /
        (
            1.0
            +
            np.exp(-calibrated_logit)
        )
    )


def run_inference(payload):

    X = validate_payload(
        payload
    )

    raw_probability = float(
        FROZEN_MODEL
        .predict_proba(X)[0, 1]
    )

    calibrated_probability = (
        calibrate_probability(
            raw_probability
        )
    )

    threshold_crossed = (
        calibrated_probability
        >=
        OPERATING_THRESHOLD
    )

    return {

        "model_id":
            MODEL_ID,

        "model_status":
            "FROZEN",

        "predictor_count":
            PREDICTOR_COUNT,

        "raw_probability":
            raw_probability,

        "calibrated_probability":
            calibrated_probability,

        "operating_point":
            OPERATING_POINT,

        "operating_threshold":
            OPERATING_THRESHOLD,

        "threshold_crossed":
            bool(threshold_crossed),

        "threshold_status":
            (
                "AT_OR_ABOVE_OP07"
                if threshold_crossed
                else
                "BELOW_OP07"
            ),

        "interpretation":
            (
                "Model risk exceeds the frozen research operating threshold."
                if threshold_crossed
                else
                "Model risk does not exceed the frozen research operating threshold."
            ),

        "diagnosis":
            "NOT PROVIDED",

        "deployment_authorization":
            "NOT AUTHORIZED",

        "warning":
            RESEARCH_WARNING
    }
