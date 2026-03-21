import os
import torch
from torchvision.utils import save_image, make_grid

def denorm(x):
    return (x + 1.0) / 2.0

def save_samples(images, path, nrow=8):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    grid = make_grid(denorm(images[:nrow * nrow].detach().cpu()), nrow=nrow)
    save_image(grid, path)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gradient_penalty(critic, real, fake, device):
    bsz = real.size(0)
    eps = torch.rand(bsz, 1, 1, 1, device=device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)

    score = critic(interp)
    grads = torch.autograd.grad(
        outputs=score,
        inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(bsz, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp