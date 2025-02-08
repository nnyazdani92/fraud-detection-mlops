from dataclasses import dataclass
from tempfile import TemporaryDirectory
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from xgboost import XGBClassifier
import mlflow


@dataclass
class FeatureSelectionResult:
    selected_features: pd.Index
    selection_mask: np.ndarray
    method_name: str


class BaseSelector:
    """Base class for feature selection methods"""

    def __init__(self):
        self.result: FeatureSelectionResult = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        raise NotImplementedError

    def log_metrics(self) -> None:
        mlflow.log_param(f"{self.result.method_name}_num_features", len(
            self.result.selected_features))


class MutualInfoSelector(BaseSelector):
    """Mutual Information feature selector"""

    def __init__(self, k=35):
        super().__init__()
        self.k = k

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        selector = SelectKBest(mutual_info_classif, k=self.k)
        selector.fit(X, y.values.ravel())

        self.result = FeatureSelectionResult(
            selected_features=X.columns[selector.get_support()],
            selection_mask=selector.get_support(),
            method_name="mi"
        )


class RFESelector(BaseSelector):
    """Recursive Feature Elimination selector"""

    def __init__(self):
        super().__init__()

    def _create_estimator(self, y: pd.DataFrame, subsample: float, device='cpu') -> XGBClassifier:
        return XGBClassifier(
            scale_pos_weight=(1 - y.values.mean()) / y.values.mean(),
            subsample=subsample,
            random_state=42,
            device=device
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, n_features=30, step=1, subsample=0.8) -> None:
        estimator = self._create_estimator(y, subsample)
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=step
        )
        selector.fit(X, y)

        self.result = FeatureSelectionResult(
            selected_features=X.columns[selector.support_],
            selection_mask=selector.support_,
            method_name="rfe"
        )


class ShapSelector(BaseSelector):
    """SHAP-based feature selector"""

    def __init__(self, n_features=25, device='cpu'):
        super().__init__()
        self.device = device
        self.n_features = n_features

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        model = XGBClassifier(
            scale_pos_weight=(1 - y.values.mean()) / y.values.mean(),
            device=self.device
        ).fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        importances = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(importances)[-self.n_features:]

        self.result = FeatureSelectionResult(
            selected_features=X.columns[top_idx],
            selection_mask=np.isin(X.columns, X.columns[top_idx]),
            method_name="shap"
        )

    def plot_summary(self, shap_values: np.ndarray, features: pd.DataFrame) -> None:
        plt.figure()
        shap.summary_plot(shap_values, features, show=False)
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "shap_summary.png"
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(path)


class FeatureVoter:
    """Ensemble feature selection voter"""

    def __init__(self, min_votes=2):
        self.min_votes = min_votes
        self.results: list[FeatureSelectionResult] = []

    def add_result(self, result: FeatureSelectionResult) -> None:
        self.results.append(result)

    def vote(self) -> FeatureSelectionResult:
        selection_matrix = pd.DataFrame(
            {r.method_name: r.selection_mask for r in self.results}
        )
        vote_counts = selection_matrix.sum(axis=1)

        final_mask = vote_counts >= self.min_votes
        return FeatureSelectionResult(
            selected_features=selection_matrix.index[final_mask],
            selection_mask=final_mask.values,
            method_name="ensemble"
        )


class FeatureSelector:
    """Main feature selection pipeline"""

    def __init__(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.X_test = X_test
        self.artifacts = FeatureSelectionArtifacts()

    def validate_data(self) -> None:
        """Validate input data quality"""
        if self.X_train.empty or self.y_train.empty:
            raise ValueError("Empty training data received")
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("Feature/target row count mismatch")

    def run_selection(self) -> tuple[dict[str, FeatureSelectionResult], FeatureSelectionResult]:
        """Execute all feature selection methods"""
        selectors = [
            MutualInfoSelector(),
            RFESelector(),
            ShapSelector()
        ]

        individual_results = {}
        for selector in selectors:
            try:
                selector.fit(self.X_train, self.y_train)
                selector.log_metrics()
                individual_results[selector.result.method_name] = selector.result
            except Exception as e:
                mlflow.log_param(
                    f"{selector.__class__.__name__}_error", str(e))
                continue

        voter = FeatureVoter()
        for result in individual_results.values():
            voter.add_result(result)

        final_result = voter.vote()
        return individual_results, final_result

    def log_selection_metrics(self, results: dict[str, FeatureSelectionResult], final_result: FeatureSelectionResult) -> None:
        """Log comprehensive selection metrics"""
        mlflow.log_metrics({
            "original_features": self.X_train.shape[1],
            "final_features_selected": len(final_result.selected_features),
            **{f"{k}_features": len(v.selected_features) for k, v in results.items()}
        })

    def create_selected_datasets(self, final_features: pd.Index) -> dict[str, pd.DataFrame]:
        """Create datasets with selected features"""

        X_train_features = self.X_train.columns[final_features]
        X_val_features = self.X_val.columns[final_features]
        X_test_features = self.X_test.columns[final_features]

        return {
            "X_train": self.X_train[X_train_features],
            "X_val": self.X_val[X_val_features],
            "X_test": self.X_test[X_test_features]
        }


class FeatureSelectionArtifacts:
    """Handles feature selection artifact logging"""

    def log_features(self, features: pd.Index) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "selected_features.parquet"
            pd.Series(features).to_frame().to_parquet(path)
            mlflow.log_artifact(
                path, "selected_features")

    def log_datasets(self, datasets: dict[str, pd.DataFrame]) -> None:
        with TemporaryDirectory() as tmp_dir:
            for i, (split, data) in enumerate(datasets.items()):
                path = Path(tmp_dir) / f"{split}_selected.parquet"
                data.to_parquet(path)
                mlflow.log_artifact(
                    path,
                    "selected"
                )
