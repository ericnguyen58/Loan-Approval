# In tests/conftest.py
import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_model():
    with patch('loan_approval.api.main.model'):
        yield