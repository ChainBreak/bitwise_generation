import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def main():
    learning_rate = 0.001
    model = SimpleLightningModule(learning_rate)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model)

class SimpleLightningModule(L.LightningModule):
    def __init__(self, learning_rate=0.001):
        super(SimpleLightningModule, self).__init__()
        self.save_hyperparameters()
        self.model = SimpleModel()
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def train_dataloader(self):
        x = torch.randn(100, 8)
        y = torch.randn(100, 8)
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

class SimpleModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=100, output_dim=8):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    main()