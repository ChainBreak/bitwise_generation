import lightning as L
from lightning.pytorch import loggers 
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def main():
    learning_rate = 0.0001
    model = SimpleLightningModule(learning_rate)
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="mps",
        logger=loggers.TensorBoardLogger("lightning_logs", name="bit_wise"),
        
        )
    trainer.fit(model)

class SimpleLightningModule(L.LightningModule):
    def __init__(self, learning_rate=0.001):
        super(SimpleLightningModule, self).__init__()
        self.save_hyperparameters()
        self.model = SimpleModel()
        self.loss_fn = nn.BCELoss()
        self.learning_rate = learning_rate

  
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]

        t = self.sample_t_for_each_sample_in_batch(x.shape)
        mask = self.sample_random_mask(t, x.shape)

        # {0,1} -> {-1,1}
        x_tri = x * 2 -1

        # Maksed bits become zero {-1,0,1}
        x_tri_masked = x_tri * mask
        # print(f"{t[0]=}\n{mask[0]=}\n{x_tri[0]=}\n{x_tri_masked[0]=}")
       
        x_hat = self(x_tri_masked)
        loss = self.loss_fn(x_hat, x)
        self.log('train_loss', loss)
        if batch_idx % 100 == 0:
            samples = self.generate_samples(1, 10)
            
        return loss
    
    def generate_samples(self, num_samples, num_steps):
        x = torch.zeros(num_samples, 8,device=self.device)

        for t,i in enumerate(torch.linspace(1, 0, num_steps)):
            t = torch.tensor([i],dtype=torch.float32)
            t = t.repeat(num_samples).to(self.device)
            mask = self.sample_random_mask(t, x.shape)
            x_tri = x * 2 -1
            x_tri_masked = x_tri * mask
            x_prob = self(x_tri_masked)
            x = x_prob > torch.rand_like(x_prob)
            x = x.float()
            print(f"{i:03}",x[0])
        return x
    
    def sample_t_for_each_sample_in_batch(self, shape: torch.Size) -> torch.Tensor:
        t = torch.rand(shape[0], device=self.device)
        return t
    
    def sample_random_mask(self, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        if not t.shape[0] == shape[0]:
            raise ValueError(f"Expected first dimension of t to be {shape[0]}, but got {t.shape[0]}")
        
        # Make sure t has the same number of dimensions as shape
        while t.dim() < len(shape):
            t = t.unsqueeze(-1)
        
        # Make binary mask with probability t
        # Each sample in the batchs gets its own mask
        mask = torch.rand(shape, device=self.device) > t

        return mask

    def train_dataloader(self):
        dataset_size = 10000

        # Generate normal random integers between 0 and 255 centered at 128
        x_int = torch.normal(128, 64, size=(dataset_size,)).clamp(0, 255).int()

        x_bit_strings = [f'{i:08b}' for i in x_int]

        x_bits = torch.tensor([list(map(float, i)) for i in x_bit_strings], dtype=torch.float32)

        dataset = TensorDataset(x_bits)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=32, 
            shuffle=True,
            num_workers=0,
        )

        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

class SimpleModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=100, output_dim=8):
        super(SimpleModel, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.sequence(x)
        return x


if __name__ == "__main__":
    main()