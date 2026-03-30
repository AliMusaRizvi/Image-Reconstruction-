import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import torchvision.transforms as T
import dataclasses
from dataclasses import dataclass
import time

#page config
st.set_page_config(
    page_title="PixelMend · Image Recovery",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# global styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── reset & base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080c12;
    color: #c8d8e8;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem 1.5rem; }

/* ── hero header ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
}
.hero-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #00e5c8;
    border: 1px solid rgba(0,229,200,0.35);
    padding: 0.3rem 0.9rem;
    border-radius: 2px;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #e8f4ff 0%, #00e5c8 60%, #0090ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}
.hero-sub {
    font-size: 1rem;
    color: #6a8499;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── upload card ── */
.upload-section { margin: 2rem 0 0; }
/* override streamlit's file uploader to look like a big drop zone */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] > div:first-child {
    background: rgba(0,229,200,0.03) !important;
    border: 2px dashed rgba(0,229,200,0.25) !important;
    border-radius: 16px !important;
    padding: 3.5rem 2rem !important;
    text-align: center !important;
    transition: border-color 0.25s, background 0.25s;
    cursor: pointer;
    min-height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
}
[data-testid="stFileUploader"] > div:first-child:hover {
    border-color: rgba(0,229,200,0.6) !important;
    background: rgba(0,229,200,0.06) !important;
}
/* hide file name row */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"] {
    display: none !important;
}
/* upload label text */
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: #6a8499 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(0,229,200,0.1) !important;
    border: 1px solid rgba(0,229,200,0.4) !important;
    color: #00e5c8 !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.2rem !important;
}
[data-testid="stFileUploader"] button:hover {
    background: rgba(0,229,200,0.2) !important;
}

/* ── drop zone icon (injected via html) ── */
.drop-hint {
    text-align: center;
    color: #3a5060;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    margin-top: 0.5rem;
}

/* ── inline mask row ── */
.mask-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.4rem 0 0.3rem;
}
.mask-row-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00e5c8;
    white-space: nowrap;
    flex-shrink: 0;
}
.mask-row-info {
    font-family: 'Space Mono', monospace;
    font-size: 0.63rem;
    color: #3a5060;
    margin-bottom: 0.8rem;
    letter-spacing: 0.06em;
}
.control-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00e5c8;
    margin-bottom: 0.6rem;
}

/* ── slider ── */
[data-testid="stSlider"] label { display: none !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    background: rgba(0,229,200,0.12) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #00e5c8 !important;
    border-color: #00e5c8 !important;
    box-shadow: 0 0 12px rgba(0,229,200,0.5) !important;
}

/* ── results section ── */
.results-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3a5060;
    margin-bottom: 0.4rem;
}
.img-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.img-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.75rem;
}
.label-original { color: #6a8499; }
.label-masked   { color: #ff8c42; }
.label-recon    { color: #00e5c8; }

/* ── progress / live reconstruction ── */
.recon-status {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #00e5c8;
    letter-spacing: 0.08em;
    text-align: center;
    padding: 0.4rem 0;
}

/* ── stats bar ── */
.stats-row {
    display: flex;
    gap: 1.5rem;
    justify-content: center;
    margin: 2rem 0 0;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(0,229,200,0.05);
    border: 1px solid rgba(0,229,200,0.2);
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    text-align: center;
    min-width: 130px;
}
.stat-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.25rem;
    color: #00e5c8;
    font-weight: 700;
}
.stat-key {
    font-size: 0.65rem;
    color: #3a5060;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}

/* ── run button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #00e5c8 0%, #0090ff 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #080c12 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    margin-top: 1rem !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── divider ── */
hr { border-color: rgba(255,255,255,0.05) !important; margin: 2.5rem 0 !important; }

/* ── progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00e5c8, #0090ff) !important;
}
</style>
""", unsafe_allow_html=True)

# Configuration
@dataclass
class MAEConfig:
    image_size: int = 224
    patch_size: int = 16
    mask_ratio: float = 0.75
    enc_dim: int = 768
    enc_depth: int = 12
    enc_heads: int = 12
    enc_mlp_ratio: float = 4.0
    dec_dim: int = 384
    dec_depth: int = 12
    dec_heads: int = 6
    dec_mlp_ratio: float = 4.0

# ── Building blocks ───────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms

class FlashSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = dropout
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2,0,3,1,4).unbind(0)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0
        )
        x = attn_out.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = FlashSelfAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio * 2 / 3)
        self.mlp   = SwiGLU(dim, hidden_dim, dropout)
        self.ls1   = nn.Parameter(1e-4 * torch.ones(dim))
        self.ls2   = nn.Parameter(1e-4 * torch.ones(dim))
    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = grid_w = grid_size
    pos_h = np.arange(grid_h, dtype=np.float32)
    pos_w = np.arange(grid_w, dtype=np.float32)
    grid  = np.stack(np.meshgrid(pos_w, pos_h), axis=0).reshape(2, -1)
    emb_dim_half = embed_dim // 2
    omega = 1.0 / (10000 ** (np.arange(emb_dim_half//2) / (emb_dim_half//2)))
    out_h = np.outer(grid[0], omega)
    out_w = np.outer(grid[1], omega)
    emb = np.concatenate([np.sin(out_h), np.cos(out_h),
                          np.sin(out_w), np.cos(out_w)], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros((1, embed_dim)), emb], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)

class MAEEncoder(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.patch_size  = cfg.patch_size
        self.num_patches = cfg.num_patches
        patch_dim = cfg.patch_size**2 * 3
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, cfg.enc_dim, bias=False),
            RMSNorm(cfg.enc_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.enc_dim))
        pos_emb = get_2d_sincos_pos_embed(cfg.enc_dim,
                    cfg.image_size // cfg.patch_size, cls_token=True)
        self.register_buffer("pos_embed", pos_emb)
        self.blocks = nn.Sequential(*[
            TransformerBlock(cfg.enc_dim, cfg.enc_heads, cfg.enc_mlp_ratio)
            for _ in range(cfg.enc_depth)
        ])
        self.norm = RMSNorm(cfg.enc_dim)
        self._init_weights()
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, patches, ids_keep):
        B, N, _ = patches.shape
        x = self.patch_embed(patches)
        x = x + self.pos_embed[:, 1:, :]
        x = torch.gather(x, 1,
            ids_keep.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, x], dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x

class MAEDecoder(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        patch_dim = cfg.patch_size**2 * 3
        self.num_patches = cfg.num_patches
        self.enc_to_dec = nn.Linear(cfg.enc_dim, cfg.dec_dim, bias=True)
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))
        pos_emb = get_2d_sincos_pos_embed(cfg.dec_dim,
                    cfg.image_size // cfg.patch_size, cls_token=True)
        self.register_buffer("dec_pos_embed", pos_emb)
        self.blocks = nn.Sequential(*[
            TransformerBlock(cfg.dec_dim, cfg.dec_heads, cfg.dec_mlp_ratio)
            for _ in range(cfg.dec_depth)
        ])
        self.norm = RMSNorm(cfg.dec_dim)
        self.pred = nn.Linear(cfg.dec_dim, patch_dim, bias=True)
        self._init_weights()
    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.pred.weight); nn.init.zeros_(self.pred.bias)
    def forward(self, latent, ids_keep, ids_restore):
        B  = latent.size(0)
        N  = ids_restore.size(1)
        nv = ids_keep.size(1)
        x  = self.enc_to_dec(latent)
        mask_tokens = self.mask_token.expand(B, N - nv, -1)
        x_vis  = x[:, 1:, :]
        x_full = torch.cat([x_vis, mask_tokens], dim=1)
        x_full = torch.gather(x_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))
        x_full = x_full + self.dec_pos_embed[:, 1:, :]
        cls    = x[:, :1, :] + self.dec_pos_embed[:, :1, :]
        x_full = torch.cat([cls, x_full], dim=1)
        x_full = self.blocks(x_full)
        x_full = self.norm(x_full)
        pred   = self.pred(x_full[:, 1:, :])
        mask   = torch.ones(B, N, device=latent.device)
        mask   = mask.scatter(1, ids_keep.to(mask.device), 0.0)
        return pred, mask

class MAE(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg     = cfg
        self.encoder = MAEEncoder(cfg)
        self.decoder = MAEDecoder(cfg)
    def _mask(self, B, device):
        N  = self.cfg.num_patches
        nv = self.cfg.num_visible
        noise       = torch.rand(B, N, device=device)
        ids_shuf    = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuf, dim=1)
        ids_keep    = ids_shuf[:, :nv]
        return ids_keep, ids_restore
    @staticmethod
    def _normalise_target(patches):
        mean = patches.mean(-1, keepdim=True)
        var  = patches.var(-1, keepdim=True)
        return (patches - mean) / (var + 1e-6).sqrt()
    def forward(self, imgs):
        B, C, H, W = imgs.shape
        patches     = patchify(imgs, self.cfg.patch_size)
        target      = self._normalise_target(patches)
        ids_keep, ids_restore = self._mask(B, imgs.device)
        latent      = self.encoder(patches, ids_keep)
        pred, mask  = self.decoder(latent, ids_keep, ids_restore)
        mse  = ((pred - target)**2 * mask.unsqueeze(-1)).sum() / (mask.sum() * pred.size(-1) + 1e-8)
        return mse, pred, mask, ids_restore
    def reconstruct(self, imgs):
        with torch.no_grad():
            loss, pred, mask, ids_restore = self.forward(imgs)
            patches   = patchify(imgs, self.cfg.patch_size)
            mean      = patches.mean(-1, keepdim=True)
            var       = patches.var(-1, keepdim=True)
            pred_denorm = pred * (var + 1e-6).sqrt() + mean
            masked    = patches.clone()
            masked[mask.bool()] = 0.5
            masked_img  = unpatchify(masked, self.cfg.patch_size, self.cfg.image_size)
            recon       = patches.clone()
            recon[mask.bool()] = pred_denorm[mask.bool()]
            recon_img   = unpatchify(recon, self.cfg.patch_size, self.cfg.image_size)
        return masked_img, recon_img, mask, pred_denorm, patches

    def reconstruct_step_by_step(self, imgs):
        """
        Three-phase generator:
          Phase 1 — patch masking reveal  (yields 'mask',   masked_img, progress, label)
          Phase 2 — encoding              (yields 'encode', None,       progress, label)
          Phase 3 — decoder block-by-step (yields 'decode', recon_img,  progress, label)
        Each yield: (phase, image_or_None, frac, status_str, block_idx, n_blocks)
        """
        with torch.no_grad():
            B, C, H, W = imgs.shape
            patches = patchify(imgs, self.cfg.patch_size)
            ids_keep, ids_restore = self._mask(B, imgs.device)
            N  = self.cfg.num_patches
            nv = ids_keep.size(1)

            # Phase 1: animate patch masking 
            mask_full = torch.ones(B, N, device=imgs.device)
            mask_full = mask_full.scatter(1, ids_keep.to(imgs.device), 0.0)
            masked_indices = mask_full[0].nonzero(as_tuple=True)[0]  # which patches get masked

            # shuffle so patches disappear in random order
            perm     = torch.randperm(masked_indices.size(0))
            shuffled = masked_indices[perm]
            n_mask   = shuffled.size(0)
            STEPS    = 20  # number of reveal steps — smoother patch-by-patch animation
            batch    = max(1, n_mask // STEPS)

            current_mask = torch.zeros(B, N, device=imgs.device)
            for step in range(STEPS):
                lo = step * batch
                hi = min(lo + batch, n_mask)
                idx = shuffled[lo:hi]
                current_mask[0, idx] = 1.0
                partial_masked = patches.clone()
                partial_masked[current_mask.bool()] = 0.5
                img_out = unpatchify(partial_masked, self.cfg.patch_size, self.cfg.image_size)
                frac = 0.02 + 0.28 * (step + 1) / STEPS
                yield ("mask", img_out, frac,
                       f"Masking patches… {int(current_mask.sum().item())} / {n_mask}",
                       step, STEPS)

            # Phase 2: encode 
            yield ("encode", None, 0.32, "Running encoder…", 0, 1)
            latent = self.encoder(patches, ids_keep)
            yield ("encode", None, 0.42, "Encoder complete — building decoder input…", 1, 1)

            # Phase 3: decoder block-by-block
            dec = self.decoder
            x   = dec.enc_to_dec(latent)
            mask_tokens = dec.mask_token.expand(B, N - nv, -1)
            x_vis  = x[:, 1:, :]
            x_full = torch.cat([x_vis, mask_tokens], dim=1)
            x_full = torch.gather(x_full, 1,
                ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))
            x_full = x_full + dec.dec_pos_embed[:, 1:, :]
            cls    = x[:, :1, :] + dec.dec_pos_embed[:, :1, :]
            x_full = torch.cat([cls, x_full], dim=1)

            n_blocks = len(dec.blocks)
            mean = patches.mean(-1, keepdim=True)
            var  = patches.var(-1, keepdim=True)

            # Yield the fully masked image as the FIRST decode frame so the
            # recon panel shows the masked state before reconstruction begins.
            masked_seed = patches.clone()
            masked_seed[mask_full.bool()] = 0.5
            masked_seed_img = unpatchify(masked_seed, self.cfg.patch_size, self.cfg.image_size)
            yield ("decode", masked_seed_img, 0.44,
                   "Starting reconstruction…", -1, n_blocks)

            for i, block in enumerate(dec.blocks):
                x_full = block(x_full)
                frac   = 0.44 + 0.56 * (i + 1) / n_blocks

                # generate partial image at every block
                partial      = dec.norm(x_full)
                partial_pred = dec.pred(partial[:, 1:, :])
                pd           = partial_pred * (var + 1e-6).sqrt() + mean
                recon        = patches.clone()
                recon[mask_full.bool()] = pd[mask_full.bool()]
                recon_img    = unpatchify(recon, self.cfg.patch_size, self.cfg.image_size)
                is_final     = (i == n_blocks - 1)
                yield ("decode", recon_img, frac,
                       f"Decoder block {i+1} / {n_blocks}",
                       i, n_blocks)

def patchify(imgs, patch_size=16):
    B, C, H, W = imgs.shape
    h = w = H // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    return x.permute(0,2,4,1,3,5).reshape(B, h*w, C*patch_size*patch_size)

def unpatchify(patches, patch_size=16, img_size=224):
    B, N, D = patches.shape
    h = w = img_size // patch_size
    C = D // (patch_size * patch_size)
    x = patches.reshape(B, h, w, C, patch_size, patch_size)
    return x.permute(0,3,1,4,2,5).reshape(B, C, img_size, img_size)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def preprocess_image(image):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0)

def denorm(t):
    m = torch.tensor(IMAGENET_MEAN).view(1,3,1,1)
    s = torch.tensor(IMAGENET_STD).view(1,3,1,1)
    return (t * s + m).clamp(0, 1)

def tensor_to_np(t):
    return denorm(t).squeeze(0).permute(1,2,0).cpu().numpy()

# mask_ratio is NOT a parameter here — weights are identical regardless of ratio.
# Masking is applied dynamically inside _mask() at inference time.
@st.cache_resource
def load_model():
    cfg = MAEConfig()
    cfg.num_patches = (cfg.image_size // cfg.patch_size) ** 2
    cfg.num_visible = int(cfg.num_patches * (1 - cfg.mask_ratio))
    cfg.num_masked  = cfg.num_patches - cfg.num_visible
    model = MAE(cfg)

    weights_path = "model_weights.pth"

    # Download from HuggingFace Hub if not present locally
    if not os.path.exists(weights_path):
        try:
            from huggingface_hub import hf_hub_download
            with st.spinner("Downloading model weights from HuggingFace…"):
                weights_path = hf_hub_download(
                    repo_id="AliMusaRizvi/mae",       
                    filename="model_weights.pth",
                )
        except Exception as e:
            st.warning(f"Could not load weights: {e}. Running with random weights.")
            model.eval()
            return model, cfg

    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, cfg


def apply_mask_ratio(cfg, mask_ratio: float):
    """Patch cfg with user-selected ratio — no model reload needed."""
    cfg = dataclasses.replace(cfg, mask_ratio=mask_ratio)
    cfg.num_visible = int(cfg.num_patches * (1 - mask_ratio))
    cfg.num_masked  = cfg.num_patches - cfg.num_visible
    return cfg

# ── hero 
st.markdown("""
<div class="hero">
  <div class="hero-badge">Masked Autoencoder · Pixel Recovery</div>
  <div class="hero-title">Restore Corrupted Images</div>
  <div class="hero-sub">
    Upload a damaged or partially corrupted image. Our MAE model will intelligently
    reconstruct missing regions using contextual understanding of the scene.
  </div>
</div>
""", unsafe_allow_html=True)

# upload + controls
uploaded_file = st.file_uploader(
    "Drop image here",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

# masking ratio — compact inline row
n_patches = (224 // 16) ** 2  # 196

col_lbl, col_sl, col_val = st.columns([1.4, 4, 0.8])
with col_lbl:
    st.markdown(
        '<div style="font-family:Space Mono,monospace;font-size:0.65rem;'
        'letter-spacing:0.18em;text-transform:uppercase;color:#00e5c8;'
        'padding-top:0.65rem;">Mask Ratio</div>',
        unsafe_allow_html=True,
    )
with col_sl:
    mask_ratio = st.slider(
        "mask_ratio",
        min_value=0.1, max_value=0.95,
        value=0.75, step=0.05,
        label_visibility="collapsed",
    )
with col_val:
    st.markdown(
        f'<div style="font-family:Space Mono,monospace;font-size:1rem;'
        f'color:#00e5c8;text-align:right;padding-top:0.55rem;font-weight:700;">'
        f'{int(mask_ratio*100)}%</div>',
        unsafe_allow_html=True,
    )

n_masked = int(n_patches * mask_ratio)
st.markdown(
    f'<div class="mask-row-info">'
    f'{n_masked} patches masked &nbsp;·&nbsp; {n_patches - n_masked} visible</div>',
    unsafe_allow_html=True,
)

# run button
run_pressed = False
if uploaded_file is not None:
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_pressed = st.button("✦  Reconstruct Image")

# live reconstruction fragment
# @st.fragment isolates this section so Streamlit flushes every st.empty()
# update to the browser immediately instead of buffering until script end.
@st.fragment
def run_reconstruction(image, mask_ratio):
    model, cfg = load_model()
    cfg = apply_mask_ratio(cfg, mask_ratio)

    st.markdown("<hr>", unsafe_allow_html=True)

    PHASE_HTML = {
        "mask":   ('<span style="font-family:Space Mono,monospace;font-size:0.6rem;'
                   'letter-spacing:0.2em;text-transform:uppercase;background:rgba(255,140,66,0.12);'
                   'color:#ff8c42;border:1px solid rgba(255,140,66,0.35);padding:0.25rem 0.7rem;'
                   'border-radius:3px;">① Masking</span>'
                   '&ensp;<span style="color:#3a5060;font-family:Space Mono,monospace;'
                   'font-size:0.6rem;">② Encoding &nbsp; ③ Decoding</span>'),
        "encode": ('<span style="color:#3a5060;font-family:Space Mono,monospace;font-size:0.6rem;">'
                   '① Masking</span>'
                   '&ensp;<span style="font-family:Space Mono,monospace;font-size:0.6rem;'
                   'letter-spacing:0.2em;text-transform:uppercase;background:rgba(0,144,255,0.12);'
                   'color:#0090ff;border:1px solid rgba(0,144,255,0.35);padding:0.25rem 0.7rem;'
                   'border-radius:3px;">② Encoding</span>'
                   '&ensp;<span style="color:#3a5060;font-family:Space Mono,monospace;'
                   'font-size:0.6rem;">③ Decoding</span>'),
        "decode": ('<span style="color:#3a5060;font-family:Space Mono,monospace;font-size:0.6rem;">'
                   '① Masking &ensp; ② Encoding</span>'
                   '&ensp;<span style="font-family:Space Mono,monospace;font-size:0.6rem;'
                   'letter-spacing:0.2em;text-transform:uppercase;background:rgba(0,229,200,0.1);'
                   'color:#00e5c8;border:1px solid rgba(0,229,200,0.35);padding:0.25rem 0.7rem;'
                   'border-radius:3px;">③ Decoding</span>'),
        "done":   ('<span style="font-family:Space Mono,monospace;font-size:0.6rem;'
                   'letter-spacing:0.2em;text-transform:uppercase;background:rgba(0,229,200,0.15);'
                   'color:#00e5c8;border:1px solid rgba(0,229,200,0.5);padding:0.25rem 0.7rem;'
                   'border-radius:3px;">✦ Complete</span>'),
    }

    # phase label placeholder
    phase_label = st.empty()

    # ── three columns — placeholders created DIRECTLY on the column object ────
    # This is the critical fix: col.empty() is reliably updated from outside
    # the `with col:` block, unlike st.empty() created inside a with context.
    col_orig, col_mask, col_recon = st.columns(3)

    with col_orig:
        st.markdown(
            '<div class="img-label label-original" style="text-align:center;'
            'margin-bottom:0.5rem;">Original</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col_mask:
        st.markdown(
            '<div class="img-label label-masked" style="text-align:center;'
            'margin-bottom:0.5rem;">Masked Input</div>', unsafe_allow_html=True)

    with col_recon:
        st.markdown(
            '<div class="img-label label-recon" style="text-align:center;'
            'margin-bottom:0.5rem;">Reconstruction</div>', unsafe_allow_html=True)

    # Placeholders declared on column objects — updates are always in-column
    mask_ph  = col_mask.empty()
    recon_ph = col_recon.empty()

    # progress bar + status
    progress_bar = st.progress(0.0)
    status_text  = st.empty()
    block_row    = st.empty()

    # seed mask panel with the original; recon panel starts blank
    input_tensor = preprocess_image(image)
    mask_ph.image(tensor_to_np(input_tensor), use_container_width=True)
    recon_ph.markdown(
        '<div style="background:#0d1820;border-radius:8px;height:224px;'
        'display:flex;align-items:center;justify-content:center;'
        'font-family:Space Mono,monospace;font-size:0.65rem;'
        'letter-spacing:0.12em;color:#1a3040;">AWAITING RECONSTRUCTION</div>',
        unsafe_allow_html=True,
    )

    # helpers
    def set_phase(p):
        phase_label.markdown(
            f'<div style="text-align:center;margin-bottom:0.6rem;">{PHASE_HTML[p]}</div>',
            unsafe_allow_html=True,
        )

    def set_status(frac, msg):
        progress_bar.progress(min(frac, 1.0))
        status_text.markdown(
            f'<div class="recon-status">⬡ {msg}</div>',
            unsafe_allow_html=True,
        )

    def draw_block_pips(done, total):
        pips = ""
        for j in range(total):
            if j < done:
                col_c, bg = "#00e5c8", "rgba(0,229,200,0.25)"
            else:
                col_c, bg = "#1a2a38", "#0d1820"
            pips += (f'<span style="display:inline-block;width:10px;height:10px;'
                     f'background:{bg};border:1px solid {col_c};border-radius:2px;'
                     f'margin:1px;"></span>')
        block_row.markdown(
            f'<div style="text-align:center;margin:0.3rem 0 0.5rem;">{pips}</div>',
            unsafe_allow_html=True,
        )

    # run the generator — every yield flushes a frame to the browser 
    prev_phase   = None
    n_dec_blocks = cfg.dec_depth

    for phase, img_t, frac, msg, blk_i, blk_n in model.reconstruct_step_by_step(input_tensor):

        if phase != prev_phase:
            set_phase(phase)
            prev_phase = phase

        set_status(frac, msg)

        if phase == "mask" and img_t is not None:
            mask_ph.image(tensor_to_np(img_t), use_container_width=True)
            block_row.empty()
            time.sleep(0.18)   # 20 steps × 0.18s ≈ 3.6s total masking animation

        elif phase == "encode":
            time.sleep(0.5)    # brief pause while encoder runs

        elif phase == "decode" and img_t is not None:
            recon_ph.image(tensor_to_np(img_t), use_container_width=True)
            if blk_i == -1:
                # seed frame — show masked patches for a full second before block 1
                time.sleep(1.2)
            else:
                draw_block_pips(blk_i + 1, blk_n)
                time.sleep(0.38)   # 12 blocks × 0.38s ≈ 4.6s total decode animation

    # done
    set_phase("done")
    progress_bar.progress(1.0)
    status_text.markdown(
        '<div class="recon-status" style="color:#00e5c8;font-size:0.8rem;">'
        '✦ Reconstruction complete</div>',
        unsafe_allow_html=True,
    )
    draw_block_pips(n_dec_blocks, n_dec_blocks)

    # ── stats ─────────────────────────────────────────────────────────────────
    n_vis  = cfg.num_visible
    n_mask = cfg.num_masked
    pct    = int(mask_ratio * 100)

    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-pill">
        <div class="stat-val">{pct}%</div>
        <div class="stat-key">Masked</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val">{n_mask}</div>
        <div class="stat-key">Patches Recovered</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val">{n_vis}</div>
        <div class="stat-key">Visible Patches</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val">16px</div>
        <div class="stat-key">Patch Size</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# trigger
if uploaded_file is not None and run_pressed:
    image = Image.open(uploaded_file).convert("RGB")
    run_reconstruction(image, mask_ratio)

elif uploaded_file is None:
    # empty-state hint
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 3rem; color: #253545;
                font-family: Space Mono, monospace; font-size: 0.75rem;
                letter-spacing: 0.1em;">
      ↑ drop an image above to begin
    </div>
    """, unsafe_allow_html=True)
