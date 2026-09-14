import os
import json
import hashlib
import pandas as pd
from typing import Dict, List, Any, Optional
from backend.app.runtime_control import get_runtime_control, RuntimeMode

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_POLICY_PATH = os.getenv("POLICY_FILE_PATH", os.path.join(BASE_DIR, "data", "generated", "causal_uplift_v10_2_predictions.csv"))
REGISTRY_PATH = os.getenv("REGISTRY_FILE_PATH", os.path.join(BASE_DIR, "data", "generated", "policy_registry.json"))

REQUIRED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "predicted_treatment_probability",
    "predicted_control_probability",
    "predicted_probability_uplift",
    "selected_policy_type",
    "selected_policy_parameter",
    "selected_policy",
    "intervention_selected",
    "oracle_control_value",
    "oracle_treatment_value",
    "oracle_incremental_value",
]

class PolicyValidationError(ValueError):
    """Raised when production policy data fails governance validation."""
    pass

class PolicyService:
    def __init__(self, csv_path: str = DEFAULT_POLICY_PATH, registry_path: str = REGISTRY_PATH):
        self.csv_path = csv_path
        self.registry_path = registry_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.lookup: Dict[str, Dict[str, Any]] = {}
        self.summary_cache: Dict[str, Any] = {}
        self.registry_metadata: Dict[str, Any] = {}
        self.hash_validated: bool = False
        self._load_and_validate()

    def _verify_sha256_hash(self) -> bool:
        if not os.path.exists(self.registry_path):
            return False
        try:
            with open(self.registry_path, "r") as f:
                self.registry_metadata = json.load(f)
            expected_hash = self.registry_metadata.get("policy_sha256")

            with open(self.csv_path, "rb") as f:
                computed_hash = hashlib.sha256(f.read()).hexdigest()

            self.hash_validated = (computed_hash == expected_hash)
            return self.hash_validated
        except Exception:
            return False

    def _load_and_validate(self):
        runtime = get_runtime_control()
        if not os.path.exists(self.csv_path):
            runtime.set_mode(RuntimeMode.FALLBACK, updated_by="POLICY_SERVICE", reason="POLICY_FILE_MISSING")
            raise PolicyValidationError(f"Production policy CSV not found at: {self.csv_path}")

        # Hash validation
        if not self._verify_sha256_hash():
            print("[RECLAIM WARNING] Policy SHA256 registry hash validation failed or registry missing! Enforcing FALLBACK runtime mode.")
            runtime.set_mode(RuntimeMode.FALLBACK, updated_by="POLICY_SERVICE", reason="POLICY_HASH_MISMATCH")

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            runtime.set_mode(RuntimeMode.FALLBACK, updated_by="POLICY_SERVICE", reason="POLICY_READ_ERROR")
            raise PolicyValidationError(f"Failed to read production policy CSV: {str(e)}")

        # 1. Validate required columns
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            runtime.set_mode(RuntimeMode.FALLBACK, updated_by="POLICY_SERVICE", reason="POLICY_SCHEMA_INVALID")
            raise PolicyValidationError(f"Missing required columns in policy data: {missing}")

        # 2. Validate row count
        total_rows = len(df)
        if total_rows != 10000:
            runtime.set_mode(RuntimeMode.FALLBACK, updated_by="POLICY_SERVICE", reason="INVALID_ROW_COUNT")
            raise PolicyValidationError(f"Expected exactly 10,000 production rows, got {total_rows}")

        # 3. Validate unique transaction IDs
        unique_txs = df["transaction_id"].nunique()
        if unique_txs != 10000:
            runtime.set_mode(RuntimeMode.FALLBACK, updated_by="POLICY_SERVICE", reason="DUPLICATE_TRANSACTION_IDS")
            raise PolicyValidationError(f"Expected 10,000 unique transaction IDs, got {unique_txs}")

        # 4. Validate selected transactions & targeting rate
        df["intervention_selected"] = df["intervention_selected"].astype(int)
        selected_count = int(df["intervention_selected"].sum())
        targeting_rate = selected_count / total_rows

        if selected_count != 500:
            runtime.set_mode(RuntimeMode.FALLBACK, updated_by="POLICY_SERVICE", reason="INVALID_TARGET_COUNT")
            raise PolicyValidationError(f"Expected exactly 500 selected transactions, got {selected_count}")

        if abs(targeting_rate - 0.05) > 1e-6:
            runtime.set_mode(RuntimeMode.FALLBACK, updated_by="POLICY_SERVICE", reason="INVALID_TARGET_RATE")
            raise PolicyValidationError(f"Expected targeting rate of 0.05 (5%), got {targeting_rate:.4f}")

        # Cleanup types for serialization
        float_cols = [
            "predicted_treatment_probability",
            "predicted_control_probability",
            "predicted_probability_uplift",
            "selected_policy_parameter",
            "oracle_control_value",
            "oracle_treatment_value",
            "oracle_incremental_value"
        ]
        for col in float_cols:
            df[col] = df[col].astype(float)

        self.df = df
        
        # Build in-memory lookup dictionary
        records = df.to_dict(orient="records")
        self.lookup = {rec["transaction_id"]: rec for rec in records}

        positive_uplift_count = int((df["predicted_probability_uplift"] > 0).sum())
        non_positive_uplift_count = total_rows - positive_uplift_count

        self.summary_cache = {
            "policy_version": "V10.2",
            "dataset_path": self.csv_path,
            "total_transactions": total_rows,
            "selected_transactions": selected_count,
            "targeting_rate": targeting_rate,
            "selected_action": "TARGET",
            "fallback_action": "RETRY_ALL",
            "positive_uplift_count": positive_uplift_count,
            "negative_zero_uplift_count": non_positive_uplift_count,
            "unique_customers": int(df["customer_id"].nunique()),
            "status": "VALIDATED" if self.hash_validated else "WARNING_HASH_UNVERIFIED",
            "governance_status": "IMMUTABLE_PRODUCTION_POLICY",
            "registry_metadata": self.registry_metadata
        }

    def get_summary(self) -> Dict[str, Any]:
        return self.summary_cache

    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        from backend.app.serializer import sanitize_production_response
        tx = self.lookup.get(transaction_id)
        return sanitize_production_response(tx, mask_ids=False) if tx else None

    def get_decision(self, transaction_id: str) -> Dict[str, Any]:
        runtime = get_runtime_control()
        if not runtime.is_active() or not self.hash_validated:
            mode = runtime.get_config().get("mode", "FALLBACK")
            reason = "POLICY_HASH_MISMATCH" if not self.hash_validated else f"RUNTIME_MODE_{mode}"
            return {
                "found": True,
                "transaction_id": transaction_id,
                "selected_policy": "RETRY_ALL",
                "intervention_selected": 0,
                "action": "RETRY_ALL",
                "fallback_applied": True,
                "fallback_reason": reason,
                "data_mode": "PRODUCTION_POLICY",
                "is_production": True,
                "decision_rationale": f"System hash unverified or runtime mode is {mode}; safe fallback RETRY_ALL applied."
            }

        tx = self.get_transaction(transaction_id)
        if not tx:
            return {
                "found": False,
                "transaction_id": transaction_id,
                "selected_policy": "RETRY_ALL",
                "intervention_selected": 0,
                "action": "RETRY_ALL",
                "fallback_applied": True,
                "fallback_reason": "TRANSACTION_NOT_FOUND",
                "data_mode": "PRODUCTION_POLICY",
                "is_production": True,
                "decision_rationale": "Transaction ID not found in V10.2 production policy; safe fallback RETRY_ALL applied."
            }
        
        return {
            "found": True,
            "transaction_id": tx["transaction_id"],
            "customer_id": tx["customer_id"],
            "selected_policy": tx["selected_policy"],
            "intervention_selected": tx["intervention_selected"],
            "action": tx["selected_policy"],
            "predicted_treatment_probability": tx["predicted_treatment_probability"],
            "predicted_control_probability": tx["predicted_control_probability"],
            "predicted_probability_uplift": tx["predicted_probability_uplift"],
            "selected_policy_type": tx["selected_policy_type"],
            "selected_policy_parameter": tx["selected_policy_parameter"],
            "uncertainty_status": "NOT_AVAILABLE",
            "score_semantics": "MODEL_RANKING_SCORE",
            "calibration_status": "NOT_CALIBRATED",
            "data_mode": "PRODUCTION_POLICY",
            "is_production": True,
            "fallback_applied": False,
            "decision_rationale": (
                "Selected for TARGET intervention based on TOP_RATE 5% causal uplift ranking."
                if tx["intervention_selected"] == 1
                else "Assigned to RETRY_ALL fallback action as transaction fell outside top 5% uplift threshold."
            ),
            "governance_note": "Production decision derived from immutable V10.2 policy dataset. No re-ranking performed."
        }

    def list_transactions(
        self,
        page: int = 1,
        limit: int = 20,
        search: Optional[str] = None,
        customer_id: Optional[str] = None,
        selected: Optional[bool] = None,
        action: Optional[str] = None,
        sort_by: Optional[str] = None,
        order: str = "asc"
    ) -> Dict[str, Any]:
        from backend.app.serializer import sanitize_production_response
        filtered_df = self.df

        if search:
            search_str = search.strip().upper()
            filtered_df = filtered_df[
                filtered_df["transaction_id"].str.upper().str.contains(search_str) |
                filtered_df["customer_id"].str.upper().str.contains(search_str)
            ]

        if customer_id:
            filtered_df = filtered_df[filtered_df["customer_id"].str.upper() == customer_id.strip().upper()]

        if selected is not None:
            val = 1 if selected else 0
            filtered_df = filtered_df[filtered_df["intervention_selected"] == val]

        if action:
            filtered_df = filtered_df[filtered_df["selected_policy"].str.upper() == action.strip().upper()]

        if sort_by and sort_by in filtered_df.columns:
            ascending = (order.lower() == "asc")
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

        total_matching = len(filtered_df)
        total_pages = max(1, (total_matching + limit - 1) // limit)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paged_records = filtered_df.iloc[start_idx:end_idx].to_dict(orient="records")
        sanitized_records = sanitize_production_response(paged_records, mask_ids=False)

        return {
            "transactions": sanitized_records,
            "pagination": {
                "page": page,
                "limit": limit,
                "total_matching": total_matching,
                "total_pages": total_pages
            }
        }

_policy_service_instance: Optional[PolicyService] = None

def get_policy_service() -> PolicyService:
    global _policy_service_instance
    if _policy_service_instance is None:
        _policy_service_instance = PolicyService()
    return _policy_service_instance
