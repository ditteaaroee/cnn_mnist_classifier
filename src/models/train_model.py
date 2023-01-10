import click
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from model import mnist_cnn

from src.data.make_dataset import mnist

#import collections


# call script 
# >>>> python src\models\train_model.py train --lr 1e-4 --epochs 1

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=5, help='number of epochs to use for training')
def train(lr, epochs):
    print(f'learning rate: {lr}')
    print(f'num. epochs: {epochs}')
    
    # Training loop
    model = mnist_cnn()
    train_set = mnist(train=True, in_folder=r'data\raw', out_folder=r'data\processed') #in_folder=r'data\raw', out_folder=r'data\processed') ????
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = []  
    for epoch in range(epochs):
        curr_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()
        
        epoch_loss = curr_loss/len(train_set)
        print(f"Epoch {epoch+1}/{epochs}. Loss: {epoch_loss}")
        losses.append(epoch_loss)
                
    torch.save(model.state_dict(), r'models\trained\trained_model.pt')

    num_epochs = list(range(1,epochs+1))
    
    plt.figure(figsize=(6,10))
    plt.plot(num_epochs, losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(r"reports\figures\training_curve.png", dpi=200)    
    # plt.figure(figsize=(6,10))
    # sns.lineplot(x=num_epochs, y=losses)
    # plt.savefig(r"reports\figures\training_curve_sns.png", dpi=200)
    
cli.add_command(train)
#cli.add_command(view_training_curve)

if __name__ == "__main__":
    cli()
