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
import time

ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # Create model with desired hyperparameters
    model = SimpleLightningModule(
        learning_rate=0.0001,
        denoise_steps=1000,
        batch_size=64,
        image_size=64,
        num_bits=8,
        log_interval_seconds=60,
    )
    
    # Load weights from checkpoint while keeping current hyperparameters
    checkpoint_path = "lightning_logs/bit_wise/version_108/checkpoints/epoch=1412-step=22608.ckpt"  # Replace with your checkpoint path
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    trainer = L.Trainer(
        max_epochs=-1,
        accelerator="mps",
        logger=loggers.TensorBoardLogger("lightning_logs", name="bit_wise"),
    )
    trainer.fit(model)

class SimpleLightningModule(L.LightningModule):
    def __init__(self, **kwargs):
        super(SimpleLightningModule, self).__init__()
        self.save_hyperparameters()

        self.model = self.create_model()

        self.loss_fn = nn.BCELoss()

        self.last_sample_time = time.time()  # Initialize with current time

    def create_model(self):
        return nn.Sequential(
            UNet(
                spatial_dims=2,
                in_channels=24,  # Input is 3 * 8 bits
                out_channels=64,  # Output is 3 * 8 bits
                channels=(64, 128, 256, 512),  # Feature channels at each layer
                strides=(2, 2, 2),  # Downsampling factors
                num_res_units=4,  # Number of residual units per layer
            ),
            nn.Conv2d(64, 24, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        x_tri = self.binary_to_ternery(x)

        # Maksed bits become zero so that the set of inputs values is {-1,0,1}
        x_tri_masked = x_tri * mask

        return self.model(x_tri_masked)

    def training_step(self, batch, batch_idx):
        p = self.hparams
        x_int = batch[0]
    
        x_bits = self.int_to_bits(x_int, p.num_bits)

        t = self.sample_t_for_each_sample_in_batch(x_bits.shape)
        mask = self.sample_random_mask(t, x_bits.shape)

        x_hat = self(x_bits, mask)
        inv_mask = 1 - mask

        # Only compute loss where the mask was 0
        loss = self.loss_fn(inv_mask * x_hat, inv_mask * x_bits)

        self.log('train_loss', loss)

        self.periodically_generate_and_log_samples()
        
        return loss
    
    def periodically_generate_and_log_samples(self):
        p = self.hparams
        current_time = time.time()
        
        # Check if a minute has passed
        if current_time - self.last_sample_time >= p.log_interval_seconds:  # 60 seconds
            self.last_sample_time = current_time
            x_bits = self.generate_bit_samples(16, p.denoise_steps)
            x_int = self.bits_to_int(x_bits, p.num_bits)
            self.log_batch_of_samples('samples', x_int)
            
    def log_batch_of_samples(self,name, x_int):
            x = x_int / 255.0
            b = x.shape[0]
            nrow = int(np.sqrt(b))
            grid = torchvision.utils.make_grid(x,nrow=nrow)
            self.logger.experiment.add_image(name, grid, self.global_step)

    def generate_bit_samples(self, num_samples, num_steps):
        print(f"Generating {num_samples} samples at epoch {self.current_epoch}")
        p = self.hparams
        x = torch.zeros(num_samples, 3*p.num_bits, p.image_size, p.image_size, device=self.device)
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
    
    def bits_to_int(self, x_bits: torch.Tensor, num_bits: int, dtype: torch.dtype = torch.uint8):
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


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    

if __name__ == "__main__":
    main()