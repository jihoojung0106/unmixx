import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
class STFTLoss(torch.nn.Module):
    def __init__(self, separate=False):
        super(STFTLoss, self).__init__()
        self.separate=separate
    def forward(self, clean_mag, mag_g, clean_pha, pha_g):
        loss_mag = F.mse_loss(clean_mag, mag_g)
        loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
        loss_pha = loss_ip + loss_gd + loss_iaf
        if not self.separate:
            return loss_mag, loss_pha
        else:
            return loss_mag, loss_ip, loss_gd, loss_iaf
def phase_losses(phase_r, phase_g):

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

class STFTLossPenalty(torch.nn.Module):
    def __init__(self, separate=False, cross_weight=0.2):
        super(STFTLossPenalty, self).__init__()
        self.separate = separate
        self.cross_weight = cross_weight

    def forward(self, clean_mag1, mag_g1, clean_pha1, pha_g1,
                      clean_mag2, mag_g2, clean_pha2, pha_g2):
        # 1. 기본 STFT 손실
        loss_mag1 = F.mse_loss(clean_mag1, mag_g1)
        loss_mag2 = F.mse_loss(clean_mag2, mag_g2)

        loss_ip1, loss_gd1, loss_iaf1 = phase_losses(clean_pha1, pha_g1)
        loss_ip2, loss_gd2, loss_iaf2 = phase_losses(clean_pha2, pha_g2)

        loss_pha1 = loss_ip1 + loss_gd1 + loss_iaf1
        loss_pha2 = loss_ip2 + loss_gd2 + loss_iaf2

        # 2. Soft mask: clean2에는 있고 clean1에는 적은 부분
        soft_mask1 = (clean_mag2 - clean_mag1).clamp(min=0)
        soft_mask2 = (clean_mag1 - clean_mag2).clamp(min=0)

        # 3. 혼입된 영역에 대해 예측된 값이 크면 패널티
        eps = 1e-8
        interference1 = torch.sum((mag_g1 * soft_mask1) ** 2) / (torch.sum(soft_mask1) + eps)
        interference2 = torch.sum((mag_g2 * soft_mask2) ** 2) / (torch.sum(soft_mask2) + eps)

        cross_loss =interference1 + interference2
        if not self.separate:
            return loss_mag1 + loss_mag2, loss_pha1 + loss_pha2, cross_loss
        else:
            return {
                'loss_mag1': loss_mag1,
                'loss_pha1': loss_pha1,
                'loss_mag2': loss_mag2,
                'loss_pha2': loss_pha2,
                'cross_loss': cross_loss,
                'interference1': interference1,
                'interference2': interference2,
                'loss_ip1': loss_ip1,
                'loss_gd1': loss_gd1,
                'loss_iaf1': loss_iaf1,
                'loss_ip2': loss_ip2,
                'loss_gd2': loss_gd2,
                'loss_iaf2': loss_iaf2,
            }
def visualize_clean_and_masks(clean_mag1, clean_mag2, mask1, mask2, filename="viz.png"):
    def to_numpy(x):
        return x.squeeze().detach().cpu().numpy()

    def safe_log(x, eps=1e-8):
        return np.log1p(x)

    mag1_np = safe_log(to_numpy(clean_mag1))
    mag2_np = safe_log(to_numpy(clean_mag2))
    mask1_np = to_numpy(mask1).astype(np.float32)
    mask2_np = to_numpy(mask2).astype(np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Clean Magnitudes (log-scaled)
    axes[0, 0].imshow(mag1_np, aspect='auto', origin='lower', cmap='magma')
    axes[0, 0].set_title('Clean Mag 1 (log)')
    axes[0, 1].imshow(mag2_np, aspect='auto', origin='lower', cmap='magma')
    axes[0, 1].set_title('Clean Mag 2 (log)')

    # Binary Masks
    axes[1, 0].imshow(mask1_np, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Interference Mask 1')
    axes[1, 1].imshow(mask2_np, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Interference Mask 2')

    for ax in axes.flat:
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Visualization saved to: {filename}")

class STFTLossPenaltyHard(torch.nn.Module):
    def __init__(self, separate=False, cross_weight=0.2, threshold_max=1.0,threshold_min=0.5):
        
        super(STFTLossPenaltyHard, self).__init__()
        self.separate = separate
        self.cross_weight = cross_weight
        self.threshold_max = threshold_max #= 1.0
        self.threshold_min = threshold_min #=0.5

    def forward(self, clean_mag1, mag_g1, clean_pha1, pha_g1,
                      clean_mag2, mag_g2, clean_pha2, pha_g2):

        # 1. 기본 STFT 손실 (magnitude + phase)
        loss_mag1 = F.mse_loss(clean_mag1, mag_g1)
        loss_mag2 = F.mse_loss(clean_mag2, mag_g2)

        loss_ip1, loss_gd1, loss_iaf1 = phase_losses(clean_pha1, pha_g1)
        loss_ip2, loss_gd2, loss_iaf2 = phase_losses(clean_pha2, pha_g2)

        loss_pha1 = loss_ip1 + loss_gd1 + loss_iaf1
        loss_pha2 = loss_ip2 + loss_gd2 + loss_iaf2

        # 2. Interference 영역 마스킹 (clean2에는 있고 clean1에는 거의 없는 부분)
        mask1 = (clean_mag2 > self.threshold_max) & (clean_mag1 < self.threshold_min)
        mask2 = (clean_mag1 > self.threshold_max) & (clean_mag2 < self.threshold_min)
        visualize=False
        if visualize:
            visualize_clean_and_masks(clean_mag1, clean_mag2, mask1, mask2, filename="interference_and_clean_log.png")
        # 3. 예측된 간섭 성분이 크면 패널티
        eps = 1e-8
        penalty1 = torch.sum((mag_g1 * mask1.float()) ** 2) / (torch.sum(mask1.float()) + eps)
        penalty2 = torch.sum((mag_g2 * mask2.float()) ** 2) / (torch.sum(mask2.float()) + eps)
        cross_loss = penalty1 + penalty2

        if not self.separate:
            return loss_mag1 + loss_mag2, loss_pha1 + loss_pha2, cross_loss
        else:
            return {
                'loss_mag1': loss_mag1,
                'loss_pha1': loss_pha1,
                'loss_mag2': loss_mag2,
                'loss_pha2': loss_pha2,
                'cross_loss': cross_loss,
                'interference1': penalty1,
                'interference2': penalty2,
                'loss_ip1': loss_ip1,
                'loss_gd1': loss_gd1,
                'loss_iaf1': loss_iaf1,
                'loss_ip2': loss_ip2,
                'loss_gd2': loss_gd2,
                'loss_iaf2': loss_iaf2,
            }