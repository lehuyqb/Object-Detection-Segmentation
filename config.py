"""
Training configuration for memory-optimized YOLO training
"""

class Config:
    # Model settings
    IMG_SIZE = 416  # Reduced from 640 to save memory
    NUM_CLASSES = 80  # Adjust based on your dataset
    
    # Training settings
    BATCH_SIZE = 4  # Reduced batch size for lower memory usage
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4
    
    # Memory optimization settings
    GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients over multiple steps
    PIN_MEMORY = False  # Disable pin_memory to reduce memory usage
    NUM_WORKERS = 2  # Reduce number of workers
    
    # CUDA memory settings
    CUDA_MEMORY_CONFIG = {
        "enable_memory_efficient_attention": True,
        "enable_flash_sdp": True,
        "enable_math": False,
        "enable_flash": True,
        "enable_mem_efficient": True,
    }
    
    # Data settings
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/val"
    LABEL_DIR = "data/labels" 