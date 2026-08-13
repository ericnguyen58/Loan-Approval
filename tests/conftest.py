"""Shared test fixtures."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_model():
    """Stand in for the trained model + thresholds, so tests can hit /predict without
    models/*.joblib existing on disk. Fixed 70% approval probability; empty thresholds
    dict means every bank falls back to predict.DEFAULT_THRESHOLD (0.5)."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    thresholds = {}

    with patch("loan_approval.models.predict._load", return_value=(model, thresholds)):
        yield model
