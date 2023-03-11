import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(model: nn.Module, data: np.ndarray, epochs: int, batch_size: int, learning_rate: float) -> nn.Module:
    """
    Trains an autoencoder model on the given data and returns the trained model.
    """
    # convert data to torch tensor
    data = torch.from_numpy(data).float()

    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model for specified number of epochs
    for epoch in range(epochs):
        # shuffle data
        indices = torch.randperm(data.shape[0])
        data = data[indices]

        # split data into batches
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size]

            # zero out gradients
            optimizer.zero_grad()

            # forward pass
            _, output = model(batch)

            # compute loss and backpropagate
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

        # print progress
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    return model

def encode_data(model: nn.Module, data: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Encodes the given data using the specified autoencoder model and returns the encoded data.
    """
    # convert data to torch tensor
    data = torch.from_numpy(data).float()

    # encode data
    encoded_data = []
    with torch.no_grad():
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size]
            encoded_batch, _ = model(batch)
            encoded_data.append(encoded_batch.numpy())
    encoded_data = np.vstack(encoded_data)

    return encoded_data

def decode_data(model: nn.Module, data: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Decodes the given encoded data using the specified autoencoder model and returns the decoded data.
    """
    # convert data to torch tensor
    data = torch.from_numpy(data).float()

    # decode data
    decoded_data = []
    with torch.no_grad():
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size]
            _, decoded_batch = model(batch)
            decoded_data.append(decoded_batch.numpy())
    decoded_data = np.vstack(decoded_data)

    return decoded_data
