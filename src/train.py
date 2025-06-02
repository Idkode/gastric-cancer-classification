import torch
from torch.utils.data import DataLoader
from torch import nn
from numpy import inf
import torch.optim as optim
import logging
from datetime import datetime
import csv

def train_model(model: nn.Module, num_epochs: int, optimizer: optim.Optimizer,
                trainloader: DataLoader, validation_loader: DataLoader, loss,
                foldername, csv_file, gpu, scheduler):

    logging.basicConfig(filename=foldername + 'train.log',
                        filemode='a',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s-%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    model = model.cuda(gpu)
    patience = num_epochs / 10
    min_val_loss = inf
    count = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in trainloader:
            labels = targets.cuda(gpu)
            images = inputs.cuda(gpu)
            outputs = model(images)
            _loss = loss(outputs, labels)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            train_loss += _loss.item() * inputs.size(0)
        train_loss /= len(trainloader)
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                labels = targets.cuda(gpu)
                images = inputs.cuda(gpu)
                outputs = model(images)
                _loss = loss(outputs, labels)  # Squeeze outputs to match target shape
                val_loss += _loss.item() * inputs.size(0)  # Accumulate loss

        val_loss /= len(validation_loader)  # Average loss over all samples
        scheduler.step(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), foldername + 'model.pth')
        else:
            count += 1
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Patience left: {patience - count}, LR: {scheduler.get_last_lr()}')
        

        with open(csv_file, 'a') as f:
            csv.writer(f).writerow([epoch + 1,
                                    train_loss,
                                    val_loss])

        if count >= patience:
            logging.info(f"Max patience reached. Last epoch: {epoch+1}")
            return 
