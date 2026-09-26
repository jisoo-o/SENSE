import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

import numpy as np
from s4_block.s4_model import S4Model


# ---------------------------------------------------------------------------
# Graph construction helpers (BioSemi 128-channel)
# ---------------------------------------------------------------------------

def _get_biosemi128_positions() -> np.ndarray:
    """Return 3-D Cartesian positions (metres) for 128 BioSemi channels via MNE."""
    import mne
    montage = mne.channels.make_standard_montage('biosemi128')
    pos_dict = montage.get_positions()['ch_pos']
    return np.array([pos_dict[name] for name in montage.ch_names], dtype=np.float32)


def _build_dense_adj(positions: np.ndarray,
                     threshold_m: float = 0.04,
                     sigma_m: float = 0.02) -> torch.Tensor:
    """Build a row-normalised dense adjacency matrix [N, N].

    Electrode pairs farther than *threshold_m* metres receive zero weight.
    Edge weight = exp(−d² / 2σ²), then row-normalised.
    """
    diff = positions[:, None] - positions[None, :]          # [N, N, 3]
    dist = np.linalg.norm(diff, axis=-1)                    # [N, N]
    adj = np.exp(-dist ** 2 / (2 * sigma_m ** 2))
    adj[dist >= threshold_m] = 0.0                          # threshold cut
    np.fill_diagonal(adj, 0.0)                              # no self-loops
    adj = adj / (adj.sum(axis=1, keepdims=True) + 1e-8)     # row-normalise
    return torch.tensor(adj, dtype=torch.float32)


def _build_random_adj(positions: np.ndarray,
                      threshold_m: float = 0.04,
                      sigma_m: float = 0.02,
                      seed: int = 42) -> torch.Tensor:
    """Random adjacency with same sparsity as topology-based adj.

    Electrode positions are randomly permuted before computing distances,
    preserving the structural density while breaking neurophysiological meaning.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(positions))
    return _build_dense_adj(positions[perm], threshold_m, sigma_m)


def _build_full_adj(n_channels: int = 128) -> torch.Tensor:
    """Fully-connected uniform adjacency matrix [N, N] (no self-loops)."""
    adj = np.ones((n_channels, n_channels), dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    adj = adj / (adj.sum(axis=1, keepdims=True) + 1e-8)     # row-normalise
    return torch.tensor(adj, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Feature-dependent GAT layer (pure PyTorch, memory-efficient)
# ---------------------------------------------------------------------------

class DenseGATLayer(nn.Module):
    """Multi-head GAT layer with feature-dependent attention, pure PyTorch.

    Attention:  e_ij = LeakyReLU(a_src · Wh_i  +  a_dst · Wh_j)
                α_ij = softmax_j(e_ij)   [masked −∞ for non-edges]
    Aggregate:  h_i' = Σ_j α_ij · Wh_j  (concat H heads → LayerNorm)

    Heads are processed **sequentially** so peak VRAM stays at [BT, C, C]
    per head instead of [BT, H, C, C] — ~1 GB fp32 vs ~4 GB for H=4.
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int = 4,
                 dropout: float = 0.3, leaky_alpha: float = 0.2):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.leaky_alpha = leaky_alpha

        # Per-head weight matrices and attention vectors
        self.W = nn.ModuleList([
            nn.Linear(d_in, self.d_head, bias=False) for _ in range(n_heads)
        ])
        self.a_src = nn.ParameterList([
            nn.Parameter(torch.empty(self.d_head)) for _ in range(n_heads)
        ])
        self.a_dst = nn.ParameterList([
            nn.Parameter(torch.empty(self.d_head)) for _ in range(n_heads)
        ])
        for k in range(n_heads):
            nn.init.xavier_uniform_(self.a_src[k].unsqueeze(0))
            nn.init.xavier_uniform_(self.a_dst[k].unsqueeze(0))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor,
                adj_weights: torch.Tensor = None) -> torch.Tensor:
        """
        h:           [BT, C, d_in]
        adj_mask:    [C, C] bool – True = no edge (→ −∞ before softmax)
        adj_weights: [C, C] float (optional) – soft structural prior,
                     multiplied into attention after softmax and renormalised.
                     Used for the ``'learnable'`` graph_type.
        Returns:     [BT, C, d_out]
        """
        head_outs = []
        adj_mask_b = adj_mask.unsqueeze(0)              # [1, C, C]
        for k in range(self.n_heads):
            Wh = self.W[k](h)                           # [BT, C, d_head]

            # Additive attention: e_ij = a_src·Wh_i + a_dst·Wh_j
            e_src = (Wh * self.a_src[k]).sum(-1)        # [BT, C]
            e_dst = (Wh * self.a_dst[k]).sum(-1)        # [BT, C]
            e = e_src.unsqueeze(-1) + e_dst.unsqueeze(-2)  # [BT, C, C]
            e = F.leaky_relu(e, self.leaky_alpha)
            e = e.masked_fill(adj_mask_b, float('-inf'))

            attn = F.softmax(e, dim=-1)                 # [BT, C, C]
            attn = torch.nan_to_num(attn, nan=0.0)      # guard isolated nodes

            # Optional learned structural prior (learnable graph_type)
            if adj_weights is not None:
                attn = attn * adj_weights.unsqueeze(0)
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            attn = self.dropout(attn)
            head_outs.append(torch.bmm(attn, Wh))       # [BT, C, d_head]

        out = torch.cat(head_outs, dim=-1)              # [BT, C, d_out]
        return self.norm(out)


# ---------------------------------------------------------------------------
# Spatial GNN encoder (memory-efficient dense implementation)
# ---------------------------------------------------------------------------

class GATSpatialEncoder(nn.Module):
    """Replace Conv1D channel-mixing with a spatial Graph Network.

    Supports two message-passing modes (controlled by ``use_gat_attention``):

    - **ResGraphConv** (default, ``use_gat_attention=False``):
      Static adjacency weights + residual linear message-passing.
      O(B·T·C²·d) memory, identical to v7.

    - **DenseGAT** (``use_gat_attention=True``):
      Feature-dependent attention via :class:`DenseGATLayer`.
      Heads processed sequentially → peak VRAM ≈ [BT, C, C] per head.

    Args:
        in_channels:        EEG channels / graph nodes (default 128).
        d_output:           Output feature dim (= embedding_size).
        d_node:             Per-node hidden dim inside the GNN.
        n_heads:            GAT attention heads.
        n_gat_layers:       Message-passing depth.
        dropout:            Dropout probability.
        graph_type:         ``'fixed'`` – frozen topology-based distance prior.
                            ``'random'`` – frozen random graph (same density, shuffled positions).
                            ``'full'`` – frozen fully-connected uniform graph.
                            ``'learnable'`` – adj weights learned from data.
        use_gat_attention:  If True, use DenseGAT; else ResGraphConv.
    """

    def __init__(self, in_channels: int = 128, d_output: int = 512,
                 d_node: int = 64, n_heads: int = 4, n_gat_layers: int = 2,
                 dropout: float = 0.3, graph_type: str = 'fixed',
                 distance_threshold_m: float = 0.04, sigma_m: float = 0.02,
                 use_gat_attention: bool = False,
                 gat_chunk_t: int = 64):
        super().__init__()
        self.n_channels = in_channels
        self.d_output = d_output
        self.graph_type = graph_type
        self.use_gat_attention = use_gat_attention
        self.gat_chunk_t = gat_chunk_t  # time steps per gradient-checkpointed chunk

        # Node input projection: scalar → d_node
        self.node_proj = nn.Sequential(
            nn.Linear(1, d_node),
            nn.GELU(),
        )

        # Message-passing layers
        if use_gat_attention:
            self.mp_layers = nn.ModuleList([
                DenseGATLayer(d_node, d_node, n_heads, dropout)
                for _ in range(n_gat_layers)
            ])
        else:
            # Residual GraphConv with static adjacency weights (v7)
            self.mp_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_node, d_node),
                    nn.LayerNorm(d_node),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_gat_layers)
            ])

        self.out_proj = nn.Linear(d_node, d_output)
        self.norm = nn.GroupNorm(1, d_output)

        # Build adjacency from BioSemi128 electrode distances
        pos = _get_biosemi128_positions()
        adj_init = _build_dense_adj(pos, distance_threshold_m, sigma_m)

        # Binary structural mask for GAT (True = no edge → −∞)
        if use_gat_attention:
            self.register_buffer('adj_mask', adj_init == 0)

        if graph_type == 'fixed':
            self.register_buffer('adj', adj_init)
        elif graph_type == 'random':
            self.register_buffer('adj', _build_random_adj(pos, distance_threshold_m, sigma_m))
        elif graph_type == 'full':
            self.register_buffer('adj', _build_full_adj(in_channels))
        else:  # 'learnable'
            eps = 1e-6
            logit_init = torch.log((adj_init + eps) / (1.0 - adj_init + eps))
            self.adj_logits = nn.Parameter(logit_init)

    # ------------------------------------------------------------------
    def _gat_chunk(self, h_flat: torch.Tensor) -> torch.Tensor:
        """Process one time-chunk: node_proj → GAT residual layers → mean-pool.

        Intended to be called via ``grad_checkpoint`` to avoid storing
        [BT_c, C, C] attention tensors during the forward pass.
        ``self.adj_mask`` / ``self.adj_logits`` are accessed directly so that
        only the EEG features ``h_flat`` need to be passed as a tensor arg.

        Args:
            h_flat: [B*Tc, C, 1]  (one chunk of the permuted EEG input)
        Returns:
            [B*Tc, d_node]  (mean-pooled node features)
        """
        h = self.node_proj(h_flat)                       # [B*Tc, C, d_node]
        adj_w = (None if self.graph_type in ('fixed', 'random', 'full')
                 else torch.sigmoid(self.adj_logits))
        for layer in self.mp_layers:
            h = h + layer(h, self.adj_mask, adj_w)       # residual
        return h.mean(dim=1)                              # [B*Tc, d_node]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG tensor [B, C, T].
        Returns:
            Spatial features [B, d_output, T].
        """
        B, C, T = x.shape

        if self.use_gat_attention:
            # Chunked + gradient-checkpointed GAT.
            # Each chunk of gat_chunk_t time steps allocates only
            # [B*chunk_t, C, C] ≈ 33 MB (vs [B*T, C, C] ≈ 918 MB full).
            out_parts = []
            for t0 in range(0, T, self.gat_chunk_t):
                x_c = x[:, :, t0: t0 + self.gat_chunk_t]   # [B, C, Tc]
                Tc = x_c.shape[2]
                h_flat = x_c.permute(0, 2, 1).reshape(B * Tc, C, 1)

                if self.training:
                    # Checkpoint recomputes [BT_c, C, C] during backward
                    # instead of storing it — saves ~900 MB VRAM.
                    pool_c = grad_checkpoint(
                        self._gat_chunk, h_flat, use_reentrant=False
                    )
                else:
                    pool_c = self._gat_chunk(h_flat)        # [B*Tc, d_node]

                proj_c = self.out_proj(pool_c)              # [B*Tc, d_output]
                out_parts.append(
                    proj_c.reshape(B, Tc, self.d_output).permute(0, 2, 1)
                )

            out = torch.cat(out_parts, dim=2)               # [B, d_output, T]
            return self.norm(out)

        else:
            # Static residual GraphConv (v7 default) — no chunking needed.
            h = x.permute(0, 2, 1).reshape(B * T, C, 1)
            h = self.node_proj(h)                           # [B*T, C, d_node]

            if self.graph_type in ('fixed', 'random', 'full'):
                adj = self.adj
            else:
                adj = torch.sigmoid(self.adj_logits)
                adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
            for layer in self.mp_layers:
                h_agg = torch.einsum('ij,bjd->bid', adj, h)
                h = h + layer(h_agg)

            out = h.mean(dim=1)
            out = self.out_proj(out)
            out = out.reshape(B, T, self.d_output).permute(0, 2, 1)
            return self.norm(out)


# BioSemi 128-channel → 10-20 nearest-neighbor mapping (computed via MNE biosemi128 montage)
# Acoustic ROI: T7→D24, T8→B14, FT7→D8, FT8→B27, TP7→D31, TP8→B11, C3→D19, C4→B22
#ACOUSTIC_PRIOR_INDICES = [119, 45, 103, 58, 126, 42, 114, 53]
ACOUSTIC_PRIOR_INDICES = [119, 45, 103, 58, 126, 42]
# N400 semantic context ROI: frontal-parietal network (N400 component peaks ~400ms post-stimulus)
# Channels: Fz(C21), FCz(C23), Cz(A1), CPz(A3), Pz(A19), CP3(A6), CP4(B3), F3(D4), F4(C4), C3(D19), C4(B22)
N400_PRIOR_INDICES = [84, 86, 0, 2, 18, 5, 34, 99, 67, 114, 53]
N400_LABELS = {
    84: 'Fz', 86: 'FCz', 0: 'Cz', 2: 'CPz', 18: 'Pz',
    5: 'CP3', 34: 'CP4', 99: 'F3', 67: 'F4', 114: 'C3', 53: 'C4',
}

# Extended prior: acoustic ROI ∪ N400 ROI (deduplicates C3/C4 shared by both)
EXTENDED_PRIOR_INDICES = list(dict.fromkeys(ACOUSTIC_PRIOR_INDICES + N400_PRIOR_INDICES))


class ChannelAttentionHead(nn.Module):
    """Learnable per-channel sigmoid gating with anatomical prior initialization."""
    def __init__(self, n_channels=128, prior_indices=None, prior_weight=2.0):
        super().__init__()
        logits = torch.zeros(n_channels)
        if prior_indices is not None:
            logits[prior_indices] = prior_weight
        self.logits = nn.Parameter(logits)

    def forward(self, x):
        # x: [B, C, T]
        weights = torch.sigmoid(self.logits)  # [C], range (0, 1)
        return x * weights.unsqueeze(0).unsqueeze(-1)


class block_conv(nn.Module):
    def __init__(self, in_channel: int=512, out_channel: int=512,
                 kernel_size: int=3, stride: int=1, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU, conv: str='conv'):
        '''A single convolution block with normaliation and dropout'''
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.padding = padding
        self.output_padding = output_padding
        self.stride = stride
        self.dropout = dropout

        if residual is not False:
            raise NotImplementedError

        if conv == 'conv':
            self.conv_layer = nn.Conv1d(in_channels=self.in_channel,
                                        out_channels=self.out_channel,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride, padding=self.padding)
        else:
            self.conv_layer = nn.ConvTranspose1d(in_channels=self.in_channel,
                                        out_channels=self.out_channel,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride, padding=self.padding,
                                        output_padding=self.output_padding)

        self.norm_layer = norm(1, out_channel)

        self.dropout_layer = nn.Dropout1d(self.dropout)

        if act_layer is not None:
            self.act_layer = act_layer()
        else:
            self.act_layer = None

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.dropout_layer(x)
        x = self.norm_layer(x)

        if self.act_layer is not None:
            x = self.act_layer(x)
        return x

class block_last_decoder(block_conv):
    def __init__(self, in_channel: int = 512, out_channel: int = 512, kernel_size: int = 3, stride: int = 1, padding: int = 1, output_padding: int = 1, norm=nn.GroupNorm, dropout: float = 0.3, residual: bool = False, act_layer=nn.GELU, conv: str = 'conv'):
        super().__init__(in_channel, out_channel, kernel_size, stride, padding, output_padding, norm, dropout, residual, act_layer, conv)

    def forward(self, x):
        x = self.conv_layer(x)
        return x

class conv_encoder_tueg(nn.Module):
    def __init__(self, n_layers: int=4, input_channels: int=19, embedding_size: int=512,
                 kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU):
        super().__init__()

        self.n_layers = n_layers

        self.block_layers = nn.ModuleList()

        self.block_layers.append(block_conv(input_channels, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer))

        for i in range(n_layers - 1):
            self.block_layers.append(block_conv(embedding_size, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer))


    def forward(self, x):
        for i, block_layer in enumerate(self.block_layers):
            x = block_layer(x)
        return x

class conv_decoder_tueg(nn.Module):
    def __init__(self, n_layers: int=4, input_channels: int=19, embedding_size: int=512,
                 kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU, is_last_layer=False):
        super().__init__()

        self.n_layers = n_layers

        self.block_layers = nn.ModuleList()

        for i in range(n_layers - 1 * is_last_layer):
            self.block_layers.append(block_conv(embedding_size, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer, 'deconv'))

        if is_last_layer:
            self.block_layers.append(block_last_decoder(embedding_size, input_channels,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, None, 'deconv'))

    def forward(self, x):
        for i, block_layer in enumerate(self.block_layers):
            x = block_layer(x)
        return x



class EEGModule(nn.Module):
    def __init__(self, n_layers_cnn: int=6,
                 use_s4=False,
                n_layers_s4: int=8,
                 device: str='cuda', embedding_size: int=512,
                 is_mask: bool=True, in_channels: int=19,
                 use_channel_attention: bool=False,
                 use_n400_prior: bool=False,
                 acoustic_prior_indices=None,
                 prior_weight: float=2.0,
                 # GNN options
                 use_gnn: bool=False,
                 graph_type: str='fixed',
                 d_node: int=64,
                 n_gat_heads: int=4,
                 n_gat_layers_gnn: int=2,
                 use_gat_attention: bool=False,
                 use_gnn_skip: bool=False,
                 skip_gate_init: float=0.0,
                 normalize_gnn_skip: bool=False):
        super().__init__()

        self.n_layers_cnn = n_layers_cnn
        self.use_s4 = use_s4
        self.n_layers_s4 = n_layers_s4
        self.device = device
        self.is_mask = is_mask
        self.use_channel_attention = use_channel_attention
        self.use_gnn = use_gnn
        self.use_gnn_skip = use_gnn_skip

        # Channel attention: acoustic-only (v5) or extended acoustic+N400 prior (v6)
        if use_channel_attention:
            if use_n400_prior:
                _prior = EXTENDED_PRIOR_INDICES
            else:
                _prior = acoustic_prior_indices if acoustic_prior_indices is not None else ACOUSTIC_PRIOR_INDICES
            self.acoustic_head = ChannelAttentionHead(in_channels, _prior, prior_weight=prior_weight)

        # Spatial encoder: GNN or Conv1D
        if use_gnn:
            self.gat_encoder = GATSpatialEncoder(
                in_channels=in_channels,
                d_output=embedding_size,
                d_node=d_node,
                n_heads=n_gat_heads,
                n_gat_layers=n_gat_layers_gnn,
                graph_type=graph_type,
                use_gat_attention=use_gat_attention,
            )
            # Skip connection: parallel CNN branch gated into GNN output.
            # Learnable scalar gate (init 0 → sigmoid=0.5) lets the model decide
            # how much CNN temporal detail to blend with GNN spatial output.
            if use_gnn_skip:
                self.cnn_skip = conv_encoder_tueg(n_layers_cnn, stride=1, padding=3, kernel_size=4, embedding_size=embedding_size, input_channels=in_channels)
                self.skip_gate = nn.Parameter(torch.full((1,), skip_gate_init))
                self.normalize_gnn_skip = normalize_gnn_skip
        else:
            self.conv_encoder = conv_encoder_tueg(n_layers_cnn, stride=1, padding=3, kernel_size=4, embedding_size=embedding_size, input_channels=in_channels)

        self.conv_encoder2 = conv_encoder_tueg(1, stride=3, padding=2, kernel_size=4, embedding_size=embedding_size, input_channels=embedding_size)

        self.deconv_encoder = conv_decoder_tueg(n_layers_cnn, stride=1, padding=3, kernel_size=4, output_padding=0, embedding_size=embedding_size, input_channels=in_channels, is_last_layer=True)
        self.deconv_encoder2 = conv_decoder_tueg(1, stride=3, padding=2, kernel_size=4, output_padding=2, embedding_size=embedding_size, input_channels=embedding_size, is_last_layer=False)

        self.s4_model = S4Model(d_input=embedding_size,
        d_output=embedding_size,
        d_model=embedding_size,
        n_layers=n_layers_s4,
        dropout=0.3,
        prenorm=False)

    @staticmethod
    def _match_length(dec: torch.Tensor, target_T: int) -> torch.Tensor:
        """Crop or zero-pad *dec* along the time axis to exactly *target_T* steps."""
        T = dec.size(2)
        if T >= target_T:
            return dec[:, :, :target_T]
        return F.pad(dec, (0, target_T - T))

    def _encode(self, x):
        """Shared encoder: spatial (Conv1D or GAT) → conv2 → S4. Returns [B, emb, T']."""
        if self.use_gnn:
            spatial_out = self.gat_encoder(x)
            if self.use_gnn_skip:
                # Parallel CNN branch: captures acoustic temporal detail.
                # CNN output may be slightly longer due to padding → crop to GNN length.
                cnn_out = self.cnn_skip(x)
                cnn_out = self._match_length(cnn_out, spatial_out.size(2))
                if self.normalize_gnn_skip:
                    cnn_std = cnn_out.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                    gnn_std = spatial_out.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                    cnn_out = cnn_out / cnn_std * gnn_std
                spatial_out = spatial_out + torch.sigmoid(self.skip_gate) * cnn_out
        else:
            spatial_out = self.conv_encoder(x)
        conv_output = self.conv_encoder2(spatial_out)
        if self.use_s4:
            mid_output = self.s4_model(conv_output.transpose(-1, -2).clone())
            mid_output = mid_output.transpose(1, 2)
        else:
            mid_output = conv_output
        return mid_output

    def forward(self, x):
        if type(x) is list:
            x = torch.cat((x[0], x[1]), dim=1)
        x = x.to(self.device)
        mask = None
        if self.is_mask:
            mask, masked_input = self.mask(x)
        else:
            masked_input = x

        if self.use_channel_attention:
            x_aud = self.acoustic_head(masked_input)
            mid_aud = self._encode(x_aud)

            # Self-reconstruction from acoustic stream
            decoder_out = self.deconv_encoder2(mid_aud.clone())
            decoder_out = self.deconv_encoder(decoder_out)

            return x, mask, mid_aud, self._match_length(decoder_out, x.size(2))
        else:
            # Single-stream path
            mid_output = self._encode(masked_input.clone())

            decoder_out = self.deconv_encoder2(mid_output.clone())
            decoder_out = self.deconv_encoder(decoder_out)

            return x, mask, mid_output, self._match_length(decoder_out, x.size(2))
