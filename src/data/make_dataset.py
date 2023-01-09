
# -*- coding: utf-8 -*-
import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import torch
import wget
from dotenv import find_dotenv, load_dotenv
from torch import Tensor
from torch.utils.data import Dataset


class mnist(Dataset):
    def __init__(self, train: bool, in_folder: str = "", out_folder: str = "") -> None:
        super().__init__()
        
        self.train = train
        self.in_folder = in_folder
        self.out_folder = out_folder
        
        if self.out_folder: # try loading from preprocessed
            try:
                self.load_preprocessed()
                return
            except ValueError: 
                print('\nPreprocessed data not found')
                self.download_data() 

        if self.train:
            content = [ ]
            for i in range(8):
                dataset = np.load(f'{in_folder}/train_{i}.npz', allow_pickle=True)
                # dataset.files to see keys
                content.append(dataset)
            data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
        else:
            content = np.load(f'{in_folder}/test.npz', allow_pickle=True)
            # dataset.files to see keys
            data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['labels'])
            
        self.data = torch.nn.functional.normalize(data)
        self.targets = targets
        
        if self.out_folder:
            self.save_preprocessed()            
        
    
    def save_preprocessed(self) -> None:
        split = "train" if self.train else "test"
        torch.save([self.data, self.targets], f"{self.out_folder}/{split}_processed.pt")
        return
    
    def load_preprocessed(self) -> None:
        split = "train" if self.train else "test"
        try:
            self.data, self.targets = torch.load(f'{self.out_folder}/{split}_processed.pt')
            print(f'Preprocessed {split} data loaded')
        except:
            raise ValueError("No preprocessed files found")
            
    def download_data(self):
        files = os.listdir(self.in_folder)
        if self.train:
            for file_idx in range(8):
                if file_idx < 5 and f"train_{file_idx}.npz" not in files:
                    print(f'\nFetching train_{file_idx} from course repo')
                    mnist_data = f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/train_{file_idx}.npz"
                    wget.download(mnist_data)
                    shutil.move(f"train_{file_idx}.npz",  f"{self.in_folder}/train_{file_idx}.npz")
                
                if file_idx > 4 and f"train_{file_idx}.npz" not in files:
                    print(f'\nFetching train_{file_idx} from course repo')
                    mnist_data_v2 = f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist_v2/train_{file_idx}.npz"
                    wget.download(mnist_data_v2)
                    shutil.move(f"train_{file_idx}.npz",  f"{self.in_folder}/train_{file_idx}.npz")
        else:
            if "test.npz" not in files:
                print(f'\nFetching test data from course repo')
                test_data = "https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/test.npz"
                wget.download(test_data)
                shutil.move("test.npz", f"{self.in_folder}/test.npz")
                    
                    
    def __len__(self) -> int:
        return self.targets.numel()
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx].float(), self.targets[idx]  


# Pre-filled
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    print('\nFetching train data')
    train = mnist(train=True, in_folder=input_filepath, out_folder=output_filepath)
    train.save_preprocessed()
    print('\n\nFetching test data')
    test = mnist(train=False, in_folder=input_filepath, out_folder=output_filepath)
    test.save_preprocessed()


    print(f'\n\nTrain data shape: {train.data.shape}')
    print(f'-- mean: {train.data.mean()}')
    print(f'-- std: {train.data.std()}')
    print(f'Train targets shape: {train.targets.shape}')
    print(f'\nTest data shape: {test.data.shape}')
    print(f'-- mean: {test.data.mean()}')
    print(f'-- std: {test.data.std()}')
    print(f'Test targets shape: {test.targets.shape}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
