import torch
from tests import _PATH_DATA
from src.data.make_dataset import mnist
import os.path

# # Skipping tests if data not exists:
# @pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
# def test_something_about_data():
    

# any test implemented needs to be wrapped in a function starting with test_xxx()
def test_data():
    train = mnist(train=True, in_folder=r'data\raw', out_folder=r'data\processed')
    assert len(train) == 40000, "Dataset did not have the correct number of samples" 
    
    test = mnist(train=False, in_folder=r'data\raw', out_folder=r'data\processed')
    assert len(test) == 5000 
    
    assert train.data.shape[1:] == torch.Size([1,28,28]) #or [728] depending on how you choose to format
    #assert len(torch.unique(test.data)) == 10 #all labels are represented