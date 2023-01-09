import click
import numpy as np
import seaborn as sns
import torch

from src.data.make_dataset import mnist
from src.models.model import mnist_cnn


@click.group()
def cli():
    pass

@click.command()
@click.argument("model_checkpoint", default=r'models\trained\trained_model.pt')
@click.option("--data_to_pred", default=r'data\processed\test_processed.pt', help='MNIST data to be classified')
def evaluate(model_checkpoint, data_to_pred):
    print(f'model_checkpoint:  {model_checkpoint}')
    print(f'folder w. images to predict:  {data_to_pred}')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loading model from pretrained
    model = mnist_cnn()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    print(model,'\n')
    
    if data_to_pred == r'data\processed\test_processed.pt': 
        # If no data is specified the preprocessed test data is predicted
        try:
            test_data = torch.load(data_to_pred)
            imgs = test_data[0]
        except ValueError:
            # No preprocessed data found. Load and process data
            print("No data found in passed directory. Loading test set from course repo.")
            test_data = mnist(train=False)
    else:
        imgs = np.load(data_to_pred)
    
    imgs = torch.tensor(imgs, dtype=torch.float, device=device)
    
    # Predictions
    log_probs = model(imgs)
    prediction = log_probs.argmax(dim=-1)
    probs = log_probs.softmax(dim=-1)
    print("Predictions")
    for i in range(imgs.shape[0]):
        print(
            f"Image {i+1} predicted to be class {prediction[i].item()} with p = {round(probs[i, prediction[i]].item(),3)}"
            )
    

cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
