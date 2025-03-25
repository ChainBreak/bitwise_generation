import lightning as L
from lightning.pytorch import loggers 
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def main():

    model = SimpleLightningModule(
        learning_rate=0.0001,
        denoise_steps=100,
        batch_size=128,
        )
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="mps",
        logger=loggers.TensorBoardLogger("lightning_logs", name="bit_wise"),
        check_val_every_n_epoch=10,
    )
    trainer.fit(model)

class SimpleLightningModule(L.LightningModule):
    def __init__(self, **kwargs):
        super(SimpleLightningModule, self).__init__()
        self.save_hyperparameters()

        self.model = SimpleModel()
        self.loss_fn = nn.BCELoss()

        self.validation_outputs = []

  
    def forward(self, x, mask):
        x_tri = self.binary_to_ternery(x)

        # Maksed bits become zero so that the set of inputs values is {-1,0,1}
        x_tri_masked = x_tri * mask

        return self.model(x_tri_masked)

    def training_step(self, batch, batch_idx):
        x = batch[0]

        t = self.sample_t_for_each_sample_in_batch(x.shape)
        mask = self.sample_random_mask(t, x.shape)

        x_hat = self(x, mask)

        loss = self.loss_fn(x_hat, x)
        self.log('train_loss', loss)


        # if batch_idx % 100 == 0:
        #     samples = self.generate_samples(1, 10)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_real = batch[0]
        
        x_pred = self.generate_samples(len(x_real), self.hparams.denoise_steps)

        x_pred_int = self.bits_to_int(x_pred)
        x_real_int = self.bits_to_int(x_real)

        self.validation_outputs.append((x_pred_int, x_real_int))

    def on_validation_epoch_end(self):
        x_pred_int = torch.cat([o[0] for o in self.validation_outputs])
        x_real_int = torch.cat([o[1] for o in self.validation_outputs])
        self.validation_outputs.clear()


        # Create histogram of predicted and real values
        plt.figure(figsize=(5, 4))
        plt.hist(x_pred_int.cpu().numpy(), bins=64, alpha=0.5, label='Predicted', density=True)
        plt.hist(x_real_int.cpu().numpy(), bins=64, alpha=0.5, label='Real', density=True)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Distribution of Predicted vs Real Values')
        plt.legend()
        self.logger.experiment.add_figure('val_distribution', plt.gcf(), self.current_epoch)
        plt.close()




    def bits_to_int(self, x):
        exponents = torch.arange(start=7,end=-1,step=-1, device=x.device)
        return torch.sum(x * (2 ** exponents), dim=1).int()



    def generate_samples(self, num_samples, num_steps):
        x = torch.zeros(num_samples, 8,device=self.device)
        with torch.no_grad():
            for t in torch.linspace(1, 0, num_steps):

                # Repeat t for each sample in the batch
                t = torch.tensor([t],dtype=torch.float32)
                t = t.repeat(num_samples).to(self.device)

                # Sample a random mask for each sample in the batch
                mask = self.sample_random_mask(t, x.shape)

                # Predict the probability of each bit being 1
                x_prob = self(x, mask)

                # Sample the bits from the probability distribution
                x_sampled = (x_prob > torch.rand_like(x_prob)).float()

                # Update the masked bits
                x = mask * x + (1 - mask) * x_sampled

                # print(f"{t[0].item():3.1f} mask",mask[0])
                # print(f"{t[0].item():3.1f} xtri",x_tri[0])

                # Print masked ternary values with consistent spacing
                # print(f"masked: {' '.join(f'{v:>3.0f}' for v in x[0])}")
        return x
    
    def binary_to_ternery(self, x):
        """Convert 0.0, 1.0 to -1.0, 1.0 which enables 0.0 to be a masked state."""
        return x * 2 - 1
    
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

        return mask.float()

    def train_dataloader(self):
        p = self.hparams
        dataset_size = 50000

        # Generate normal random integers between 0 and 255 centered at 128
        x_int = torch.normal(128, 32, size=(dataset_size,)).clamp(0, 255).int()

        x_bit_strings = [f'{i:08b}' for i in x_int]

        x_bits = torch.tensor([list(map(float, i)) for i in x_bit_strings], dtype=torch.float32)

        dataset = TensorDataset(x_bits)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=p.batch_size, 
            shuffle=True,
            num_workers=0,
        )

        return dataloader

    def val_dataloader(self):
        return self.train_dataloader()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
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