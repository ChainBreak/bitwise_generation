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
import diffusers

ssl._create_default_https_context = ssl._create_unverified_context

def main():

    # Load weights from checkpoint while keeping current hyperparameters
    checkpoint_path = "lightning_logs/bit_wise/version_137/checkpoints/epoch=8696-step=139152.ckpt"
    model = SimpleLightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        learning_rate=0.0001,
        denoise_steps=100,
        batch_size=64,
        image_size=64,
        num_bits=8,
        log_interval_seconds=60,
    )
    
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

    def create_model(self) -> diffusers.UNet2DModel:

        return diffusers.UNet2DModel(
            in_channels=3*self.hparams.num_bits,
            out_channels=3*self.hparams.num_bits,
            down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels = (64, 128, 256, 512),
            layers_per_block = 1,
        )

    def forward(self, x, t):

        return F.sigmoid(self.model(x, timestep=t).sample)

    def training_step(self, batch, batch_idx):
        p = self.hparams
        x_int = batch[0]
    
        x_bits = self.int_to_bits(x_int, p.num_bits)

        t = self.sample_t_for_each_sample_in_batch(x_bits.shape)

        noisy_x_bits = self.sample_random_x(x_bits, t)

        x_prob = self(noisy_x_bits, t)

        loss = self.loss_fn(x_prob, x_bits)

        self.log('train_loss', loss)

        self.periodically_generate_and_log_samples()
        
        return loss

            
    def sample_random_x(self, x_prob: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        # Make sure t has the same number of dimensions as shape
        while t.dim() < len(x_prob.shape):
            t = t.unsqueeze(-1)
        
        # t=0 blend_prob = x_prob
        # t=1 blend_prob = 0.5 (fully random)
        blend_prob = (1-t)*x_prob + t*0.5

        # Make binary mask with probability t
        # Each sample in the batchs gets its own mask
        x = torch.rand_like(blend_prob) < blend_prob

        return x.float()

    
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

        x_steps = []

        with torch.no_grad():
            for t in torch.linspace(1, 0, num_steps):

                # Repeat t for each sample in the batch
                t = torch.tensor([t],dtype=torch.float32)
                t = t.repeat(num_samples).to(self.device)

                x = self(x,t)

                x = self.sample_random_x(x, t)

                x_int = self.bits_to_int(x, p.num_bits)
                x_steps.append(x_int[0])

        x_steps = torch.stack(x_steps, dim=0)
        self.log_batch_of_samples('x_steps', x_steps)


        return x
    
    def binary_to_ternery(self, x):
        """Convert 0.0, 1.0 to -1.0, 1.0 which enables 0.0 to be a masked state."""
        return x * 2 - 1
    
    def sample_t_for_each_sample_in_batch(self, shape: torch.Size) -> torch.Tensor:
        t = torch.rand(shape[0], device=self.device)
        return t


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
        
        # convert to gray encoding
        x_int = x_int ^ (x_int >> 1)

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

        # convert gray encoding back to binary
        mask = x_int.clone()
        for i in range(1,num_bits+1):
            x_int ^= (mask >> i)

        return x_int


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    

if __name__ == "__main__":
    main()