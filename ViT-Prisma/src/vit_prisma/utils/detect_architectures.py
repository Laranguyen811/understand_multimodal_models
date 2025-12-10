def detect_architecture(model):
    '''
    Comprehensive architecture detection covering all major AI architectures. 
    Based on HuggingFace landscape as of 2024-2025.
    '''

    model_class = model.__class__.__name__
    module_name = model.__class__.__module__
    model_class_lower = model_class.lower() # Lowercase names of the model, making detection case-insensitive
    # ======= Transformer Architectures =======
    # BERT family 
    if 'bert' in model_class_lower:
        if 'roberta' in model_class_lower:
            return 'roberta'
        elif 'deberta' in model_class_lower:
            return 'deberta'
        elif 'albert' in model_class_lower:
            return 'albert'
        elif 'distilbert' in model_class_lower:
            return 'distilbert'
        return 'bert'
    
    # GPT family
    if 'gpt' in model_class_lower:
        if 'gpt2' in model_class_lower:
            return 'gpt2'
        elif 'gptneo' in model_class_lower:
            return 'gpt_neo'
        elif 'gptj' in model_class_lower:
            return 'gptj'
        return 'gpt_generic'
    
    # T5
    if 't5' in model_class_lower:
        if 'flan' in model_class_lower:
            return 'flan_t5'
        return 't5'
    
    #LLaMA
    if 'llama' in model_class_lower:
        return 'llama'
    
    # Mistral
    if 'mistral' in model_class_lower or 'mixtral' in model_class_lower:
        return 'mistral'
    
    # BART
    if 'bart' in model_class_lower:
        return 'bart'
    
    # Claude
    if 'claude' in model_class_lower:
        return 'claude'
    
    # Falcon 
    if 'falcon' in model_class_lower:
        return 'falcon'
    
    # MPT 
    if 'mpt' in model_class_lower:
        return 'mpt'
    
    # Bloom
    if 'bloom' in model_class_lower:
        return 'bloom'
    
    # OPT
    if 'opt' in model_class_lower:
        return 'opt'
    
    # ======= Vision Transformers =======
    # ViT
    if 'vit' in model_class_lower or 'visiontransformer' in model_class_lower:
        if 'deit' in model_class_lower:
            return 'deit'
        elif 'beit' in model_class_lower:
            return 'beit'
        return 'vit'
    
    # Swin
    if 'swin' in model_class_lower:
        return 'swin'
    
    # ======= Multimodal Models =======

    # CLIP
    if 'clip' in model_class_lower:
        if hasattr(model, 'vision_model'):
            return 'clip_vision'
        elif hasattr(model, 'text_model'):
            return 'clip_text'
        return 'clip'
    
    # BLIP
    if 'blip' in model_class_lower:
        return 'blip'
    
    # LLaVA
    if 'llava' in model_class_lower:
        return 'llava'
    
    # ALIGN
    if 'align' in model_class_lower:
        return 'align'
    
    # Flamingo
    if 'flamingo' in model_class_lower:
        return 'flamingo'
    
    # GPT-4V 
    if 'gpt4v' or 'gpt-4v' in model_class_lower:
        return 'gpt4v'

    # Gemini
    if 'gemini' in model_class_lower:
        return 'gemini'

    # CoCa
    if 'coca' in model_class_lower:
        return 'coca' 
    
    # ======= Generative Models =======

    # Diffusion Models
    if 'diffusion' or 'unet2d' in model_class_lower:
        if 'stable' or 'sdxl' in model_class_lower:
            return 'stable_diffusion'
        return 'diffusion'
    
    # GANs
    if 'gan' in model_class_lower:
        if 'stylegan' in model_class_lower:
            return 'stylegan'
        elif 'biggan' in model_class_lower:
            return 'biggan'
        return 'gan'
    
    # VAE
    if 'vae' or 'variationalautoencoder' in model_class_lower:
        return 'vae'
    
    # Normalizing Flows
    if 'flow' or 'normalizingflow' in model_class_lower:
        return 'normalizing_flow'
    
    # DALL-E
    if 'dalle' in model_class_lower:
        return 'dalle'
    
    if 'imagen' in model_class_lower:
        return 'imagen'
    

    
    # ======= Traditional Convolutional Neural Networks (CNNs) =======

    # ResNet
    if 'resnet' in model_class_lower:
        return 'resnet'
    
    # ConvNeXt (modern CNN with transformer inspiration)
    if 'convnext' in model_class_lower:
        return 'convnext'
    
    # EfficientNet
    if 'efficientnet' in model_class_lower:
        return 'efficientnet'
    
    # VGG
    if 'vgg' in model_class_lower:
        return 'vgg'
    
    # Inception 
    if 'inception' in model_class_lower:
        return 'inception'
    
    # MobileNet
    if 'mobilenet' in model_class_lower:
        return 'mobilenet'
    
    # ======= Recurrent Architectures =======

    # LSTM
    if 'lstm' in model_class_lower:
        return 'lstm'
    
    # GRU
    if 'gru' in model_class_lower:
        return 'gru'
    
    # ======= Specialised Architectures =======
    
    # Graphe Neural Networks
    if 'gnn' or'graph' in model_class_lower:
        if 'graphormer' in model_class_lower:
            return 'graphormer'
        elif 'gcn' in model_class_lower:
            return 'gcn'
        elif 'gat' in model_class_lower:
            return 'gat'
        return 'gnn'
    
    # Reinforcement Learning
    
    if 'dqn' in model_class_lower:
        return 'dqn'
    elif 'a2c' or 'actorcritic' in model_class_lower:
        return 'a2c'
    elif 'ppo' in model_class_lower:
        return 'ppo'
    elif 'sac' in model_class_lower:
        return 'sac'
    
    # Memory Networks
    if 'memory' and 'network' in model_class_lower:
        return 'memory_network'
    
    # Capsule Networks
    if 'capsule' or 'capsnet' in model_class_lower:
        return 'capsnet'
    
    # ======= Emerging Hybrid Architectures =======
    
    # Mamba (state space models)
    if 'mamba' in model_class_lower:
        return 'mamba'

    # RWKV
    if 'rwkv' in model_class_lower:
        return 'rwkv'

    # Jamba (Transformer-Mamba hybrid)
    if 'jamba' in model_class_lower:
        return 'jamba'

    # Neural ODE
    if 'ode' or 'neuralode' in model_class_lower:
        return 'neural_ode'

    # Perceiver
    if 'perceiver' in model_class_lower:
        return 'perceiver'

    # Retention Networks
    if 'retnet' or 'retentive' in model_class_lower:
        return 'retnet'

    # ======= Audio Models =======
    # Whisper
    if 'whisper' in model_class_lower:
        return 'whisper'
    
    # Wav2Vec2
    if 'wav2vec2' in model_class_lower:
        return 'wav2vec2'
    
    # HuBERT
    if 'hubert' in model_class_lower:
        return 'hubert'
    
    # ======= MLP =======
    # Multi-Layer Perceptron (often as part of larger models)
    if 'mlp' in model_class_lower and len(model_class_lower) < 20: # Avoid matching MLPBlock in transformers
        return 'mlp'
    
    # ======= Fallback Detection =======
    # TransformerLens (interpretability library)
    if hasattr(model, 'cfg') and hasattr(model, 'blocks'):
        return 'transformerlens'
    
    # Generic transformer with GPT-style layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return 'gpt_generic'
    
    # Generic transformer with BERT-style layers
    if hasattr(model,'encoder') and hasattr(model.encoder, 'layer'):
        return 'bert_generic'
    
    # Vision Encoder
    if hasattr(model, 'vision_model') or hasattr(model, 'visual'):
        return 'vision_generic'
    
    # Text encoder
    if hasattr(model, 'text_model') or hasattr(model, 'language_model'):
        return 'text_generic'
    
    # CNN detection
    if any(hasattr(model, attr) for attr in ['conv1', 'features', 'layer1']):
        return 'cnn_generic'
    
    raise ValueError(
        f"Unknown architecture: {model_class_lower}\n"
        f"Module: {module_name}\n"
        f"Consider adding detection for this model type."
    )
    
# Architecture categories for easier querying

ARCHITECTURE_CATEGORIES = {
    'transformer_encoder': ['bert', 'roberta', 'deberta', 'albert', 'distilbert'],
    'transformer_decoder': ['gpt2', 'gpt_neo', 'gpt_j', 'llama', 'mistral'],
    'transformer_encoder_decoder': ['t5', 'flan_t5', 'bart'],
    'vision_transformer': ['vit', 'deit', 'beit', 'swin'],
    'multimodal': ['clip', 'blip', 'llava', 'align'],
    'generative': ['stable_diffusion', 'gan', 'stylegan', 'biggan', 'vae', 'normalizing_flow'],
    'cnn': ['resnet', 'convnext', 'efficientnet', 'vgg', 'inception', 'mobilenet'],
    'recurrent': ['lstm', 'gru'],
    'graph': ['gnn', 'graphormer', 'gcn', 'gat'],
    'rl': ['dqn', 'a2c', 'ppo', 'sac'],
    'hybrid': ['mamba', 'rwkv', 'jamba', 'neural_ode', 'perceiver', 'retnet'],
    'audio': ['whisper', 'wav2vec2', 'hubert'],
}

def get_architecture_category(arch):
    '''
    Get the category of an architecture.

    '''
    for category, archs in ARCHITECTURE_CATEGORIES.items():
        if arch in archs:
            return category
    return 'other'
def get_architecture_info(model):
    '''
    Gets the comprehensive architecture information.

    '''
    arch = detect_architecture(model)
    category = get_architecture_category(arch)

    info = {
        'arch': arch,
        'category': category,
    }

    # Get layer/head counts
    if hasattr(model, 'config'):
        config = model.config
        for attr in ['num_hidden_layers', 'n_layer', 'num_layers']:
            if hasattr(config,attr):
                info['n_layers'] = getattr(config, attr)
                break
    
        for attr in ['num_attention_heads', 'n_head', 'num_heads']:
            if hasattr(config, attr):
                info['n_heads'] = getattr(config, attr)
                break
        
        if hasattr(config, 'hidden_size'):
            info['hidden_size'] = config.hidden_size

        
        # TransformerLens
        if hasattr(model, 'cfg'):
            info['n_layers'] = model.cfg.n_layers
            info['n_heads'] = model.cfg.n_heads
            info['hidden_size'] = model.cfg.d_model
        
        # Manual counting for some architectures
        if 'n_layers' not in info:
            if hasattr(model,'transformer') and hasattr(model.transformer, 'h'):
                info['n_layers'] = len(model.transformer.h)
            elif hasattr(model,'encoder'):
                if hasattr(model.encoder, 'layer'):
                    info['n_layers'] = len(model.encoder.layer)
                elif hasattr(model.encoder, 'layers'):
                    info['n_layers'] = len(model.encoder.layers)
        
        return info

    
        
