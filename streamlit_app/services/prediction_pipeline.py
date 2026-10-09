
"""
Sepsis AI — Frozen End-to-End Prediction Pipeline

Research prototype only.
Not authorized for clinical deployment.

Pipeline:
longitudinal history
    -> frozen feature engineering
    -> exact 230-predictor latest-hour representation
    -> frozen B03 inference
    -> frozen calibration
    -> frozen OP07 threshold.

No model fitting, recalibration, threshold optimization,
or Test-driven development is permitted here.
"""

from pathlib import Path

from streamlit_app.services.feature_engineering import (
    engineer_frozen_features,
    latest_prediction_rows,
)

from streamlit_app.services.frozen_inference import (
    run_inference,
)


def run_longitudinal_prediction(
    longitudinal_df,
    approved_feature_registry_path
):
    """
    Run one governed prediction for the latest observation
    of exactly one patient trajectory.
    """

    patient_count = (
        longitudinal_df["PatientID"]
        .nunique()
    )

    if patient_count != 1:
        raise ValueError(
            "Prototype prediction requires exactly one "
            "patient trajectory at a time."
        )

    engineered_history, predictor_matrix = (
        engineer_frozen_features(
            longitudinal_df,
            approved_feature_registry_path
        )
    )

    latest_matrix = (
        latest_prediction_rows(
            engineered_history,
            predictor_matrix
        )
    )

    if len(latest_matrix) != 1:
        raise RuntimeError(
            "Latest-hour extraction did not produce "
            "exactly one prediction row."
        )

    latest_features = (
        latest_matrix
        .iloc[0]
    )

    prediction = (
        run_inference(
            latest_features
        )
    )

    return {
        "patient_id":
            str(
                engineered_history[
                    "PatientID"
                ]
                .iloc[-1]
            ),

        "prediction_hour":
            float(
                engineered_history[
                    "ICULOS"
                ]
                .iloc[-1]
            ),

        "observed_history_hours":
            int(
                len(
                    engineered_history
                )
            ),

        "predictor_count":
            int(
                latest_matrix.shape[1]
            ),

        "raw_probability":
            float(
                prediction[
                    "raw_probability"
                ]
            ),

        "calibrated_probability":
            float(
                prediction[
                    "calibrated_probability"
                ]
            ),

        "operating_threshold":
            float(
                prediction[
                    "operating_threshold"
                ]
            ),

        "threshold_status":
            prediction[
                "threshold_status"
            ],

        "diagnosis":
            "NOT PROVIDED",

        "deployment_authorization":
            "NOT AUTHORIZED",

        "research_use_only":
            True,
    }
