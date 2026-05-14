from vit_prisma.vjepa_hf.modeling_vjepa import VJEPAConfig

CONFIGS = {
    "vjepa_v1_vit_huge": VJEPAConfig(
            crop_size=256,
            uniform_power=True,
            hidden_size=1280,
            num_attention_heads=16,
            num_hidden_layers=32,
            layer_norm_eps=1e-6,
            use_sdpa=True,
        ),
    "vjepa_v1_vit_huge_384": VJEPAConfig(
            crop_size=384,
            uniform_power=True,
            hidden_size=1280,
            num_attention_heads=16,
            num_hidden_layers=32,
            layer_norm_eps=1e-6,
            use_sdpa=True,
        ),
    "vjepa_v1_vit_giant_xformers_rope": VJEPAConfig(
            crop_size=256,
            uniform_power=True,
            hidden_size=1408,
            num_attention_heads=22,
            num_hidden_layers=40,
            use_rope=True,
            mlp_ratio=48 / 11,
            use_sdpa=True,
        ),

}
