import torch
import torch.nn.functional as F

class DiffusionStrategy:
    def prepare_batch(self, batch, device):
        raise NotImplementedError

    def compute_loss(self, model, x_input, t, noise_target, extra_info=None):
        raise NotImplementedError

class JointDistributionStrategy(DiffusionStrategy):
    def prepare_batch(self, batch, device):
        # batch shape: [B, L, N, F]
        b, l, n, f = batch.shape
        
        # Joint Mode: Flatten N*F
        x_flat = batch.view(b, l, n * f).to(device)
        
        return x_flat

    def compute_loss(self, model, x_input, t, noise_target, extra_info=None):
        # x_input means x_noisy
        noise_pred = model(x_input, t)
        
        # MSE loss fn
        loss = F.mse_loss(noise_pred, noise_target)

        return loss, {"mse": loss.item()}


# Condition DDPM (Future work)
class ConditionalStrategy(DiffusionStrategy):
    def prepare_batch(self, batch, device):
        pass

    def compute_loss(self, model, x_input, t, noise_target, extra_info=None):
        pass