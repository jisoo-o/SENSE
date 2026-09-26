import torch
import torch.nn as nn
from torch.nn import functional as F

import commons


def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses

def eeg_loss(eeg, eeg_hat, eeg_lengths):
  cossim_loss_fn = torch.nn.CosineSimilarity(dim=2)

  mask_from_lengths = torch.zeros(eeg.size(), device=eeg.device)

  for item_i in range(mask_from_lengths.size(0)):
    mask_from_lengths[item_i,:,:eeg_lengths[item_i]] = 1

  masked_eeg = mask_from_lengths * eeg
  masked_eeg_hat = mask_from_lengths * eeg_hat
  cossim_loss = cossim_loss_fn(masked_eeg, masked_eeg_hat)

  cossim_loss = 1 - torch.sum(cossim_loss) / (eeg.size(0) * eeg.size(1)) # divide by batch size * channel size


  return cossim_loss


class InfoNCELoss(nn.Module):
  """InfoNCE contrastive loss with FIFO memory bank of CLIP embeddings."""
  def __init__(self, embed_dim=512, bank_size=256, temperature=0.07):
    super().__init__()
    self.temperature = temperature
    self.bank_size = bank_size
    self.register_buffer('bank', F.normalize(torch.randn(bank_size, embed_dim), dim=-1))
    self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))

  @torch.no_grad()
  def _enqueue(self, clip_embs):
    batch_size = clip_embs.size(0)
    ptr = int(self.ptr)
    if ptr + batch_size <= self.bank_size:
      self.bank[ptr:ptr + batch_size] = clip_embs.detach()
    else:
      overflow = (ptr + batch_size) - self.bank_size
      self.bank[ptr:] = clip_embs[:batch_size - overflow].detach()
      self.bank[:overflow] = clip_embs[batch_size - overflow:].detach()
    self.ptr[0] = (ptr + batch_size) % self.bank_size

  def forward(self, projected, clip_embs):
    """
    projected:  [B, 512] — projected EEG embedding (z_aud projection)
    clip_embs:  [B, 512] — CLIP text embedding
    Returns: scalar InfoNCE loss
    """
    proj_norm = F.normalize(projected, dim=-1)
    clip_norm = F.normalize(clip_embs, dim=-1)

    # In-batch similarity: [B, B]
    logits_batch = torch.mm(proj_norm, clip_norm.t()) / self.temperature
    # Memory bank similarity: [B, bank_size] (detach bank from graph)
    logits_bank = torch.mm(proj_norm, self.bank.clone().detach().t()) / self.temperature
    # Concatenate: [B, B + bank_size], positive at diagonal of first B columns
    logits = torch.cat([logits_batch, logits_bank], dim=1)

    labels = torch.arange(proj_norm.size(0), device=proj_norm.device)
    loss = F.cross_entropy(logits, labels)

    self._enqueue(clip_norm)
    return loss


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  shorter_len = min(z_p.size(2), m_p.size(2))
  z_p = z_p[:,:,:shorter_len]
  m_p = m_p[:, :, :shorter_len]
  logs_p = logs_p[:,:,:shorter_len]
  logs_q = logs_q[:,:,:shorter_len]
  z_mask = z_mask[:,:,:shorter_len]

  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l
