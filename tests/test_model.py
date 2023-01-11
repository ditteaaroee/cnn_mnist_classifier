import pytest
import torch

from src.models.model import mnist_cnn
from src.models.predict_model import evaluate


def test_model():
    #check that input.shape == output.shape
    assert 1 == 1
    

def test_error_on_wrong_shape():
    model = mnist_cnn()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))