class Config:
    encoder: str = "ViT-B-16-plus-240"
    decoder: str = "ai-forever/rugpt3medium_based_on_gpt2"
    batch_size: int = 24
    num_epochs: int = 100
    frozen_gpt: int = 20
    frozen_clip: int = 60
    learning_rate: float = 2e-4
    save_path: str = "model_saves/"
    prefix_length: int = 20
    only_prefix: int = False
    prefix: str = "prefix_small"
    device: str = "cuda:0"
    save_every: int = 1
    warmup_steps: int = 2000