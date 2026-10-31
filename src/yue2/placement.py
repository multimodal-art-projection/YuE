"""Exclusive AR/NAR residency; callers must serialize access to the model."""
from .devices import empty_cache


def ar_modules(model):
    yield model.model.embed_tokens
    yield model.lm_head
    for layer in model.model.layers:
        yield from (layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp)


def nar_modules(model):
    yield from (model.vae2llm, model.llm2vae, model.time_embedder, model.latent_pos_embed)
    for layer in model.model.layers:
        yield from (layer.nar_input_layernorm, layer.nar_self_attn, layer.nar_pre_mlp_layernorm, layer.nar_mlp)


def activate_phase(model, phase, device):
    """Evict the inactive experts before moving the active experts and shared norm."""
    active, inactive = (ar_modules, nar_modules) if phase == "ar" else (nar_modules, ar_modules)
    for module in inactive(model):
        module.to(device="cpu")
    empty_cache(device)
    for module in active(model):
        module.to(device=device)
    model.model.norm.to(device=device)
    model.model.rotary_emb.to(device=device)
