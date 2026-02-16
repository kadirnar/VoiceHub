from typing import List

import torch

from voicehub.models.echo.sampling import (
    KVCache,
    _concat_kv_caches,
    _multiply_kv_cache,
    _temporal_score_rescale,
)
from voicehub.models.echo.model import EchoDiT


@torch.inference_mode()
def sample_blockwise_euler_cfg_independent_guidances(
    model: EchoDiT,
    speaker_latent: torch.Tensor,
    speaker_mask: torch.Tensor,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    rng_seed: int,
    block_sizes: List[int],
    num_steps: int,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_min_t: float,
    cfg_max_t: float,
    truncation_factor: float | None,
    rescale_k: float | None,
    rescale_sigma: float | None,
    speaker_kv_scale: float | None,
    speaker_kv_max_layers: int | None,
    speaker_kv_min_t: float | None,
    continuation_latent: torch.Tensor | None = None,
) -> torch.Tensor:

    INIT_SCALE = 0.999  # so that we can apply rescale to first step

    device, dtype = model.device, model.dtype
    batch_size = text_input_ids.shape[0]

    rng = torch.Generator(device=device).manual_seed(rng_seed)

    t_schedule = torch.linspace(1., 0., num_steps + 1, device=device) * INIT_SCALE

    text_mask_uncond = torch.zeros_like(text_mask)
    speaker_mask_uncond = torch.zeros_like(speaker_mask)

    kv_text_cond = model.get_kv_cache_text(text_input_ids, text_mask)
    kv_speaker_cond = model.get_kv_cache_speaker(speaker_latent.to(dtype))

    # masks prevent decoder from attending to unconds:
    kv_text_full = _concat_kv_caches(kv_text_cond, kv_text_cond, kv_text_cond)
    kv_speaker_full = _concat_kv_caches(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)

    full_text_mask = torch.cat([text_mask, text_mask_uncond, text_mask], dim=0)
    full_speaker_mask = torch.cat([speaker_mask, speaker_mask, speaker_mask_uncond], dim=0)

    prefix_latent = torch.zeros((batch_size, sum(block_sizes) , 80), device=device, dtype=torch.float32)

    start_pos = 0
    if continuation_latent is not None:
        continuation_len = continuation_latent.shape[1]
        prefix_latent = torch.cat([continuation_latent, prefix_latent], dim=1)
        start_pos = continuation_len

    for block_size in block_sizes:
        if speaker_kv_scale is not None:
            _multiply_kv_cache(kv_speaker_cond, speaker_kv_scale, speaker_kv_max_layers)
            kv_speaker_full = _concat_kv_caches(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)

        full_prefix_latent = torch.cat([prefix_latent, prefix_latent, prefix_latent], dim=0)
        kv_latent_full = model.get_kv_cache_latent(full_prefix_latent.to(dtype))
        kv_latent_cond = [(k[:batch_size], v[:batch_size]) for k, v in kv_latent_full]

        x_t = torch.randn((batch_size, block_size, 80), device=device, dtype=torch.float32, generator=rng)
        if truncation_factor is not None:
            x_t = x_t * truncation_factor

        for i in range(num_steps):
            t, t_next = t_schedule[i], t_schedule[i + 1]

            has_cfg = ((t >= cfg_min_t) * (t <= cfg_max_t)).item()

            if has_cfg:
                v_cond, v_uncond_text, v_uncond_speaker = model(
                    x=torch.cat([x_t, x_t, x_t], dim=0).to(dtype),
                    t=(torch.ones((batch_size * 3,), device=device) * t).to(dtype),
                    text_mask=full_text_mask,
                    speaker_mask=full_speaker_mask,
                    start_pos=start_pos,
                    kv_cache_text=kv_text_full,
                    kv_cache_speaker=kv_speaker_full,
                    kv_cache_latent=kv_latent_full,
                ).float().chunk(3, dim=0)
                v_pred = v_cond + cfg_scale_text * (v_cond - v_uncond_text) + cfg_scale_speaker * (v_cond - v_uncond_speaker)
            else:
                v_pred = model(
                    x=x_t.to(dtype),
                    t=(torch.ones((batch_size,), device=device) * t).to(dtype),
                    text_mask=text_mask,
                    speaker_mask=speaker_mask,
                    start_pos=start_pos,
                    kv_cache_text=kv_text_cond,
                    kv_cache_speaker=kv_speaker_cond,
                    kv_cache_latent=kv_latent_cond,
                ).float()

            # optional temporal score rescaling: https://arxiv.org/pdf/2510.01184
            if rescale_k is not None and rescale_sigma is not None:
                v_pred = _temporal_score_rescale(v_pred, x_t, t, rescale_k, rescale_sigma)

            # optional kv speaker scaling
            if speaker_kv_scale is not None and t_next < speaker_kv_min_t and t >= speaker_kv_min_t:
                _multiply_kv_cache(kv_speaker_cond, 1. / speaker_kv_scale, speaker_kv_max_layers)
                kv_speaker_full = _concat_kv_caches(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)

            x_t = x_t + v_pred * (t_next - t)

        prefix_latent[:, start_pos:start_pos + block_size] = x_t
        start_pos += block_size

    return prefix_latent


if __name__ == "__main__":
    import torchaudio
    from voicehub.models.echo.sampling import (
        load_model_from_hf,
        load_fish_ae_from_hf,
        load_pca_state_from_hf,
        load_audio,
        get_text_input_ids_and_mask,
        get_speaker_latent_and_mask,
        ae_encode,
        ae_decode,
        crop_audio_to_flattening_point,
    )

    model = load_model_from_hf()
    fish_ae = load_fish_ae_from_hf()
    pca_state = load_pca_state_from_hf()


    # example 1, generate 320 in three blocks

    speaker_audio_path = "/path/to/speaker/audio.wav"
    speaker_audio = load_audio(speaker_audio_path).cuda()
    speaker_latent, speaker_mask = get_speaker_latent_and_mask(fish_ae, pca_state, speaker_audio)

    text = "[S1] Alright, I'm going to demo this new model called Echo TTS."
    text_input_ids, text_mask = get_text_input_ids_and_mask([text], max_length=None, device="cuda")

    latent_out = sample_blockwise_euler_cfg_independent_guidances(
        model=model,
        speaker_latent=speaker_latent,
        speaker_mask=speaker_mask,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        rng_seed=0,
        block_sizes=[128, 128, 64], # (sums to 320, so will be ~15 seconds; supports up to 640)
        num_steps=40,
        cfg_scale_text=3.0,
        cfg_scale_speaker=5.0,
        cfg_min_t=0.5,
        cfg_max_t=1.0,
        truncation_factor=0.8,
        rescale_k=None,
        rescale_sigma=None,
        speaker_kv_scale=None,
        speaker_kv_max_layers=None,
        speaker_kv_min_t=None,
    )
    audio_out = ae_decode(fish_ae, pca_state, latent_out)
    audio_out = crop_audio_to_flattening_point(audio_out, latent_out[0])
    torchaudio.save("output_blockwise.wav", audio_out[0].cpu(), 44100)



    # ___________________________________________________________
    # example 2: with continuation latent (use same speaker audio as first example, generate from partial output of first example)

    continuation_audio_path = "output_blockwise.wav" # can be any path
    continuation_audio = load_audio(continuation_audio_path).cuda()
    continuation_latent, continuation_mask = get_speaker_latent_and_mask(fish_ae, pca_state, continuation_audio)

    continuation_latent = continuation_latent[:, :continuation_mask.sum()]

    text = "[S1] Alright, I'm going to demo this new model called Echo TTS, and now, we're going to continue from the audio we already generated and add some more text." 
    # NOTE this MUST include the text from the continuation prefix. can use https://huggingface.co/jordand/whisper-d-v1a to get in-distribution transcription automatically.

    text_input_ids, text_mask = get_text_input_ids_and_mask([text], max_length=None, device="cuda")

    continuation_block_sizes = [256] # (generate up to 12 more seconds)
    # NOTE: these do not include the continuation latent length, so sum(block_sizes) + continuation_latent.shape[1] should be < 640 (to be in-distribution with training data)

    latent_out_continued = sample_blockwise_euler_cfg_independent_guidances(
        model=model,
        speaker_latent=speaker_latent,
        speaker_mask=speaker_mask,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        rng_seed=0,
        block_sizes=continuation_block_sizes,
        num_steps=40,
        cfg_scale_text=3.0,
        cfg_scale_speaker=3.0,
        cfg_min_t=0.5,
        cfg_max_t=1.0,
        truncation_factor=0.8,
        rescale_k=None,
        rescale_sigma=None,
        speaker_kv_scale=None,
        speaker_kv_max_layers=None,
        speaker_kv_min_t=None,
        continuation_latent=continuation_latent,
    )
    audio_out_continued = ae_decode(fish_ae, pca_state, latent_out_continued)
    audio_out_continued = crop_audio_to_flattening_point(audio_out_continued, latent_out_continued[0])
    torchaudio.save("output_blockwise_continued.wav", audio_out_continued[0].cpu(), 44100)
