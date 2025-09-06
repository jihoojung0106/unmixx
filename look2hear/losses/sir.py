import torch
def project(source, target, eps=1e-8):
    # source, target: (B, T)
    scale = torch.sum(source * target, dim=-1, keepdim=True) / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    return scale * target

def sir_loss(ests, targets, eps=1e-8):
    """
    ests: (B, 2, T) - predicted separated sources
    targets: (B, 2, T) - ground truth sources

    Returns: scalar loss (lower = better separation)
    """
    est0 = ests[:, 0, :]  # (B, T)
    est1 = ests[:, 1, :]
    ref0 = targets[:, 0, :]
    ref1 = targets[:, 1, :]

    # projection of est0 onto ref1 (interference)
    interf0 = project(est0, ref1)  # (B, T)
    target0 = project(est0, ref0)

    # projection of est1 onto ref0 (interference)
    interf1 = project(est1, ref0)
    target1 = project(est1, ref1)

    # SIR = target_energy / interference_energy
    num0 = torch.sum(target0 ** 2, dim=-1)
    denom0 = torch.sum(interf0 ** 2, dim=-1) + eps
    sir0 = 10 * torch.log10(num0 / denom0 + eps)

    num1 = torch.sum(target1 ** 2, dim=-1)
    denom1 = torch.sum(interf1 ** 2, dim=-1) + eps
    sir1 = 10 * torch.log10(num1 / denom1 + eps)

    # maximize SIR → minimize negative
    sir = (sir0 + sir1) / 2
    return -sir.mean()
