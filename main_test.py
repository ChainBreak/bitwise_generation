import main
import torch

import pytest

@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_int_to_bit(device):
    
    model = main.SimpleLightningModule()
    b,c,h,w = 8,3,4,4
    image_int = torch.tensor(list(range(b*c*h*w))).reshape(b,c,h,w).to(device)

    image_bits = model.int_to_bits(image_int, 9)
 
    image_int_reconstructed = model.bits_to_int(image_bits, 9, dtype=torch.int16)

    assert torch.all(image_int == image_int_reconstructed)
