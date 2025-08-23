"""Configuration class for training settings."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Model parameters
    model_name: str = "Qwen/Qwen1.5-3B"
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 1000
    warmup_steps: int = 100
    
    # Data parameters
    train_file: Path = Path("data/train.json")
    valid_file: Optional[Path] = None
    max_length: int = 512
    
    # Output parameters
    output_dir: Path = Path("checkpoints")
    logging_steps: int = 10
    save_steps: int = 100
    
    # Hardware parameters
    use_8bit: bool = True
    use_4bit: bool = False
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    
    def __post_init__(self) -> None:
        """Convert string paths to Path objects and create directories."""
        if isinstance(self.train_file, str):
            self.train_file = Path(self.train_file)
        if isinstance(self.valid_file, str):
            self.valid_file = Path(self.valid_file)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
