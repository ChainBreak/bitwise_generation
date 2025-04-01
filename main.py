import pathlib
import lightning as L
from lightning.pytorch import loggers 
import numpy as np
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ssl
from monai.networks.nets import UNet

ssl._create_default_https_context = ssl._create_unverified_context

def main():

    model = SimpleLightningModule(
        learning_rate=0.00001,
        denoise_steps=100,
        batch_size=32,
        image_size=64,
        )

    trainer = L.Trainer(
        max_epochs=200,
        accelerator="mps",
        logger=loggers.TensorBoardLogger("lightning_logs", name="bit_wise"),
        check_val_every_n_epoch=10,
    )
    trainer.fit(model)

class SimpleLightningModule(L.LightningModule):
    def __init__(self, **kwargs):
        super(SimpleLightningModule, self).__init__()
        self.save_hyperparameters()

        self.model = UNet(
            spatial_dims=2,
            in_channels=24,  # Input is 3 * 8 bits
            out_channels=24,  # Output is 3 * 8 bits
            channels=(16, 32, 64, 128),  # Feature channels at each layer
            strides=(2, 2, 2),  # Downsampling factors
            num_res_units=2,  # Number of residual units per layer
        )
        self.loss_fn = nn.BCELoss()

        self.validation_outputs = []

  
    def forward(self, x, mask):
        x_tri = self.binary_to_ternery(x)

        # Maksed bits become zero so that the set of inputs values is {-1,0,1}
        x_tri_masked = x_tri * mask

        return self.model(x_tri_masked)

    def training_step(self, batch, batch_idx):
        x_int = batch[0]
        x_bits = self.int_to_bits(x_int, 8)

        t = self.sample_t_for_each_sample_in_batch(x_bits.shape)
        mask = self.sample_random_mask(t, x_bits.shape)

        x_hat = self(x_bits, mask)
        inv_mask = 1 - mask

        # Only compute loss where the mask was 0
        loss = self.loss_fn(inv_mask * x_hat, inv_mask * x_bits)

        self.log('train_loss', loss)
           
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     x_real = batch[0]
        
    #     x_pred = self.generate_samples(len(x_real), self.hparams.denoise_steps)

    #     x_pred_int = self.bits_to_int(x_pred)
    #     x_real_int = self.bits_to_int(x_real)

    #     self.validation_outputs.append((x_pred_int, x_real_int))


    def on_validation_epoch_end(self):
        x_pred_int = torch.cat([o[0] for o in self.validation_outputs])
        x_real_int = torch.cat([o[1] for o in self.validation_outputs])
        self.validation_outputs.clear()

        x_int = np.arange(0, 256)
        x_bits = self.int_to_bits(x_int).to(self.device)
        mask = torch.ones_like(x_bits)

        x_bit_prob = 1.0 - (x_bits - self(x_bits, mask)).abs()
        x_int_prob = torch.prod(x_bit_prob, dim=1)

        # Create histogram of predicted and real values
        plt.figure(figsize=(5, 4))
        plt.hist(x_pred_int.cpu().numpy(), bins=64, alpha=0.5, label='Predicted', density=True)
        plt.hist(x_real_int.cpu().numpy(), bins=64, alpha=0.5, label='Real', density=True)
        plt.plot(x_int, x_int_prob.cpu().numpy(), label='Ideal', color='black')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Distribution of Predicted vs Real Values')
        plt.legend()
        self.logger.experiment.add_figure('val_distribution', plt.gcf(), self.current_epoch)
        plt.close()




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
        current_folder = pathlib.Path(__file__).resolve().parent
        data_root = current_folder / '.dataset_root'

        # Define your training dataset and dataloader here
        train_dataset = torchvision.datasets.Flowers102(
            root=data_root,
            split="train",
            download=True, 
            transform = self.transform,
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=p.batch_size,
            shuffle=True,
            )
        
        return train_loader
    
    def transform(self, pil_image):
        p = self.hparams
        transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(min(pil_image.size)),
            torchvision.transforms.Resize((p.image_size , p.image_size)),
            torchvision.transforms.PILToTensor(),
        ])
        return transform(pil_image)
    
    def int_to_bits(self, x_int: torch.Tensor, num_bits: int):
        b,c,h,w = x_int.shape
        
        # Move channels into batch dimension
        x_int = x_int.reshape(b*c,h,w)
        x_bits = torch.zeros(b*c,num_bits,h,w, device=x_int.device, dtype=torch.float32)

        for i in range(num_bits):
            x_bits[:,i,:,:] = (x_int >> i) & 1

        x_bits = x_bits.reshape(b,c*num_bits,h,w)

        return x_bits
    
    def bits_to_int(self, x_bits: torch.Tensor, num_bits: int, dtype: torch.dtype = torch.int8):
        # Calculate the number of integer channels
        b,c_bits,h,w = x_bits.shape
        c = c_bits // num_bits

        # Convert float bits to boolean
        x_bits = x_bits > 0.5

        x_bits = x_bits.reshape(b*c,num_bits,h,w)
        x_int = torch.zeros(b*c,h,w, device=x_bits.device, dtype=dtype)

        for i in range(num_bits):
            x_int += x_bits[:,i,:,:] * (2 ** i)

        x_int = x_int.reshape(b,c,h,w)

        return x_int

   

    def val_dataloader(self):
        return self.train_dataloader()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    

if __name__ == "__main__":
    main()