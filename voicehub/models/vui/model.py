from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from voicehub.models.vui.config import Config
from voicehub.models.vui.fluac import Fluac, FluacConfig
from voicehub.models.vui.patterns import DelayedPatternProvider
from voicehub.models.vui.rope import apply_rotary_emb, precompute_freqs_cis
from voicehub.models.vui.tok import CustomByT5Tokenizer
from voicehub.models.vui.utils import load_what_you_can
from voicehub.optimization.protocols import OptimizationCompileTarget


class KVCache(nn.Module):
    """Fixed-size key/value buffer for autoregressive inference with in-place
    updates."""

    def __init__(
        self,
        batch_size: int,
        max_seqlen: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        cache_shape = (batch_size, n_kv_heads, max_seqlen, head_dim)

        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        # input_pos: (T,), k_val: (B, nh, T, d)
        assert input_pos.size(0) == k_val.size(-2)

        k_out = self.k_cache
        v_out = self.v_cache
        input_pos = input_pos.int()
        k_out[:, :, input_pos] = k_val.to(k_out.dtype)
        v_out[:, :, input_pos] = v_val.to(k_out.dtype)

        return k_out, v_out


def repeat_kv(x: torch.Tensor, n_reps: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA expansion)."""
    bs, n_kv_heads, T, head_dim = x.shape

    return (
        x[:, :, None, :, :].expand(bs, n_kv_heads, n_reps, T,
                                   head_dim).reshape(bs, n_kv_heads * n_reps, T, head_dim))


class MHA(nn.Module):
    """Multi-Head Attention with optional grouped-query attention and rotary
    embeddings."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        *,
        block_idx: int,
        bias: bool = False,
        dropout: float = 0.0,
        causal: bool = False,
        use_rotary_emb: bool = True,
    ):
        super().__init__()

        head_dim = dim // n_heads

        self.use_rotary_emb = use_rotary_emb
        self.block_idx = block_idx
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.causal = causal
        if n_heads % n_kv_heads:
            raise ValueError("`n_heads` must be divisible by `n_kv_heads`.")
        self.n_reps = n_heads // n_kv_heads
        qkv_dim = (n_heads + 2 * n_kv_heads) * head_dim
        self.Wqkv = nn.Linear(dim, qkv_dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.kv_cache = None

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor | None = None,
        input_pos: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ):
        B, T, _ = x.size()

        dropout_p = self.dropout if self.training else 0.0

        qkv = self.Wqkv(x)
        if self.n_heads == self.n_kv_heads:
            qkv = (qkv.reshape(B, T, 3, self.n_heads, self.head_dim).permute(0, 2, 3, 1, 4))
            q, k, v = qkv.unbind(dim=1)  # (B, h, T, d)
        else:
            q, k, v = torch.split(
                qkv,
                [
                    self.head_dim * self.n_heads,
                    self.head_dim * self.n_kv_heads,
                    self.head_dim * self.n_kv_heads,
                ],
                dim=-1,
            )
            q = q.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.use_rotary_emb:
            q = apply_rotary_emb(freqs_cis, q)
            k = apply_rotary_emb(freqs_cis, k)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        if self.n_reps > 1:
            k = repeat_kv(k, self.n_reps)
            v = repeat_kv(v, self.n_reps)

        # A caller-provided mask owns causality as well as padding. PyTorch's
        # SDPA rejects/varies across versions when both mechanisms are active.
        is_causal = (self.causal and self.kv_cache is None and attn_mask is None)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=is_causal,
            attn_mask=attn_mask,
        )

        out = self.out_proj(out.permute(0, 2, 1, 3).reshape(B, T, self.dim))

        return out


class MLP(nn.Module):
    """Standard two-layer feed-forward with configurable activation and
    dropout."""

    def __init__(self, *, d_model: int, bias: bool, dropout: float, act=nn.GELU, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.act = act()
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class LlamaMLP(nn.Module):
    """SwiGLU-style MLP following the LLaMA architecture."""

    def __init__(self, *, d_model: int, multiple_of: int = 256, bias: bool = False, **kwargs) -> None:
        super().__init__()
        hidden_dim = 4 * d_model
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Block(nn.Module):
    """Pre-norm transformer block with causal self-attention and a LLaMA-style
    MLP."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        block_idx: int,
        bias: bool,
        dropout: float,
        norm_eps: float = 1e-5,  # use 1e-6 for rms
        use_rotary_emb: bool = True,
    ):
        super().__init__()

        self.block_idx = block_idx
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = MHA(
            d_model,
            n_heads,
            n_kv_heads,
            block_idx=block_idx,
            bias=bias,
            dropout=dropout,
            causal=True,
            use_rotary_emb=use_rotary_emb,
        )
        self.mlp_norm = RMSNorm(d_model, eps=norm_eps)
        self.mlp = LlamaMLP(d_model=d_model, bias=bias, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor | None = None,
        input_pos: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ):
        x = x + self.attn(
            self.attn_norm(x),
            freqs_cis=freqs_cis,
            input_pos=input_pos,
            attn_mask=attn_mask,
        )
        x = x + self.mlp(self.mlp_norm(x))

        return x


class Decoder(nn.Module):
    """Stack of transformer blocks with RoPE and optional KV-cache for
    inference."""

    def __init__(
        self,
        *,
        n_layers: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        bias: bool,
        dropout: float,
        max_seqlen: int = 4096,
        rope_theta: float = 10000.0,
        rope_theta_rescale_factor: float = 1.0,
        norm_eps: float = 1e-5,
        use_rotary_emb: bool = True,
        rope_dim: int | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.use_rotary_emb = use_rotary_emb

        self.max_seqlen = max_seqlen
        self.blocks = nn.ModuleList([
            Block(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                block_idx=block_idx,
                bias=bias,
                dropout=dropout,
                norm_eps=norm_eps,
                use_rotary_emb=use_rotary_emb,
            ) for block_idx in range(n_layers)
        ])
        self.norm = RMSNorm(d_model, eps=norm_eps)

        self.attn_mask = None

        head_dim = d_model // n_heads

        rope_dim = rope_dim or head_dim

        assert rope_dim <= head_dim  # apply RoPE to a fraction of embeddings

        freqs_cis = precompute_freqs_cis(
            rope_dim,
            max_seqlen,
            theta=rope_theta,
            theta_rescale_factor=rope_theta_rescale_factor,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def allocate_inference_cache(self, batch_size: int, device: str, dtype=torch.bfloat16):
        for block in self.blocks:
            block.attn.kv_cache = KVCache(
                batch_size, self.max_seqlen, block.n_kv_heads, block.head_dim, dtype).to(device)

        # I don't understand why this is needed
        self.attn_mask = torch.tril(
            torch.ones(self.max_seqlen, self.max_seqlen, dtype=torch.bool, device=device))

    def deallocate_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = None

        self.attn_mask = None

    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        attn_mask: Tensor | None = None,
    ):
        if self.use_rotary_emb:
            freqs_cis = self.freqs_cis[input_pos]
            if input_pos.ndim == 2:
                freqs_cis = freqs_cis.unsqueeze(1)
        else:
            freqs_cis = None

        if attn_mask is None and self.attn_mask is not None:
            attn_mask = self.attn_mask[None, None, input_pos]

        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis, input_pos=input_pos, attn_mask=attn_mask)

        x = self.norm(x)

        return x


class Vui(nn.Module):
    """Vui text-to-speech model: byte-level text encoder + multi-codebook audio
    decoder."""

    BASE = "vui-100m-base.pt"
    COHOST = "vui-cohost-100m.pt"
    ABRAHAM = "vui-abraham-100m.pt"

    def __init__(
        self,
        config: Config | None = None,
        *,
        codec: Fluac | None = None,
    ):
        super().__init__()
        config = Config() if config is None else config
        if not isinstance(config, Config):
            raise TypeError("`config` must be a Vui Config or None.")
        self.codec = Fluac.from_pretrained() if codec is None else codec
        self.config = config
        cfg = config.model
        self.tokenizer = CustomByT5Tokenizer()
        self.use_rotary_emb = cfg.use_rotary_emb
        self.token_emb = nn.Embedding(self.tokenizer.vocab_size, cfg.d_model)

        self.pattern_provider = DelayedPatternProvider(n_q=cfg.n_quantizers)

        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(cfg.codebook_size + 8, cfg.d_model) for _ in range(cfg.n_quantizers)])

        n_kv_heads = cfg.n_heads

        max_seqlen = cfg.max_text_tokens + cfg.max_audio_tokens
        self.decoder = Decoder(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=n_kv_heads,
            bias=cfg.bias,
            dropout=cfg.dropout,
            max_seqlen=max_seqlen + cfg.n_quantizers,
            rope_dim=cfg.rope_dim,
            rope_theta=cfg.rope_theta,
            rope_theta_rescale_factor=cfg.rope_theta_rescale_factor,
        )

        self.audio_heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.codebook_size + 8, bias=cfg.bias) for _ in range(cfg.n_quantizers)])

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose only quality-safe Vui compile boundaries.

        Real-checkpoint tests changed autoregressive generation for both
        compiler-default and explicit dynamic inference policies.
        Inference therefore exposes no compile target. Training retains
        its full- sequence decoder boundary, which does not use mutable
        generation caches or waveform reconstruction.
        """
        if mode == "training":
            return (OptimizationCompileTarget(
                "decoder.forward",
                self.decoder,
                "forward",
            ), )
        if mode == "inference":
            return ()
        raise ValueError("Vui compile targets require 'inference' or 'training' mode.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def from_pretrained(
        checkpoint_path: str | dict = ABRAHAM,
        *,
        codec_path: str | Path | None = None,
        codec: Fluac | None = None,
        model_config: Config | Mapping[str, object] | None = None,
        codec_config: FluacConfig | Mapping[str, object] | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        verify_official_integrity: bool = True,
        **config_kwargs,
    ):
        native_artifact = None
        if isinstance(checkpoint_path, dict):
            if model_config is not None or codec_config is not None:
                raise TypeError(
                    "Do not pass native graph configurations with a legacy "
                    "in-memory Vui checkpoint.")
            checkpoint = checkpoint_path
        else:
            direct_checkpoint = Path(checkpoint_path).expanduser()
            is_native_directory = (
                direct_checkpoint.is_dir() and (direct_checkpoint / "model.safetensors").is_file())
            is_native_checkpoint = (
                direct_checkpoint.is_file() and direct_checkpoint.suffix.lower() == ".safetensors")
            if is_native_directory or is_native_checkpoint:
                from voicehub.models.vui.checkpoint import load_vui_safetensors, resolve_native_vui_artifact

                native_artifact = resolve_native_vui_artifact(direct_checkpoint, )
                if (is_native_checkpoint and direct_checkpoint.resolve() != native_artifact.model_checkpoint):
                    raise ValueError(
                        "Vui.from_pretrained() requires the model Safetensors "
                        "file, not the codec component.")
                checkpoint_path = native_artifact.model_checkpoint
                if codec_path is None:
                    codec_path = native_artifact.codec_checkpoint
                if model_config is None:
                    model_config = native_artifact.model_config
                if codec_config is None:
                    codec_config = native_artifact.codec_config
                state_dict = load_vui_safetensors(
                    checkpoint_path,
                    component="model",
                )
                checkpoint = None
            elif codec_path is not None and direct_checkpoint.is_file():
                checkpoint_path = direct_checkpoint.resolve()
            else:
                from voicehub.models.vui.artifacts import resolve_vui_artifacts

                artifacts = resolve_vui_artifacts(
                    checkpoint_path,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                    verify_official_integrity=verify_official_integrity,
                )
                checkpoint_path = artifacts.model_checkpoint
                if codec_path is None:
                    codec_path = artifacts.codec_checkpoint
            if native_artifact is None:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
        if codec is not None and codec_path is not None:
            raise TypeError("Pass either `codec` or `codec_path`, not both.")
        if codec is None:
            if codec_path is None:
                from voicehub.models.vui.artifacts import VUI_CODEC_FILENAME, VUI_REPO_ID, VUI_REVISION

                codec = Fluac.from_pretrained(
                    VUI_CODEC_FILENAME,
                    repo_id=VUI_REPO_ID,
                    revision=revision or VUI_REVISION,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                )
            else:
                codec = Fluac.from_pretrained(
                    str(codec_path),
                    config=codec_config,
                )

        if native_artifact is None:
            config_values = checkpoint["config"]
            state_dict = checkpoint["model"]
        elif isinstance(model_config, Config):
            config_values = model_config.model_dump()
        elif isinstance(model_config, Mapping):
            config_values = dict(model_config)
        else:  # pragma: no cover - guarded by native artifact validation
            raise TypeError("Native Vui model configuration must be a mapping.")
        config = Config.from_dict({
            **config_values,
            **config_kwargs,
        })

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("text_embedding.", "token_emb."): v for k, v in state_dict.items()}
        model = Vui(config, codec=codec)
        if native_artifact is None:
            load_what_you_can(state_dict, model)
        else:
            model_state = {
                name: value
                for name, value in model.state_dict().items() if not name.startswith("codec.")
            }
            missing = sorted(set(model_state) - set(state_dict))
            unexpected = sorted(set(state_dict) - set(model_state))
            if missing or unexpected:
                raise ValueError(
                    "Native Vui model Safetensors do not match the declared "
                    f"graph (missing={missing!r}, unexpected={unexpected!r}).")
            model.load_state_dict(
                {
                    **model.state_dict(),
                    **state_dict,
                },
                strict=True,
            )
        return model

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        wrapper_config=None,
    ):
        """Export a standalone native Vui + Fluac Safetensors directory."""
        from voicehub.models.vui.checkpoint import export_vui_pretrained

        return export_vui_pretrained(
            self,
            save_directory,
            wrapper_config=wrapper_config,
        )

    @staticmethod
    def from_pretrained_inf(
        checkpoint_path: str | dict,
        **config_kwargs,
    ):
        return Vui.from_pretrained(checkpoint_path, **config_kwargs).eval()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
