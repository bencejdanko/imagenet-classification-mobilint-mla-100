import os

class Config:
    BASE_DIR = '/content'
    IMAGE_ROOT = os.path.join(BASE_DIR, 'imagenet_train20a')
    TRAIN_LIST = os.path.join(BASE_DIR, 'imagenet_train20.txt')
    VAL_IMAGE_ROOT = os.path.join(BASE_DIR, 'imagenet_val20')
    VAL_LIST = os.path.join(BASE_DIR, 'imagenet_val20.txt')

    BATCH_SIZE = 1024
    NUM_CLASSES = 20
    INPUT_SHAPE = (240, 240)
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    
    # Training variations
    AUG_MODE = 'mixup' # Options: 'none', 'mixup', 'cutmix', 'fmix', 'resizemix', 'hmix'
    
    # WandB and Checkpointing
    WANDB_PROJECT = 'resnet10-imagenet-20'
    WANDB_ENTITY = 'bdanko' # Set this to your wandb username if needed
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    SAVE_MODEL = True

    @classmethod
    def to_table(cls):
        """
        Returns a string representation of the config hyperparameters in a table format.
        """
        header = f"{'Hyperparameter':<20} | {'Value':<20}"
        separator = "-" * 43
        lines = [header, separator]
        
        # Filter out built-in attributes, methods, and file paths
        exclude_keywords = ['ROOT', 'LIST', 'PATH']
        attrs = {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (classmethod, staticmethod)) 
                and not callable(v)
                and not any(kw in k for kw in exclude_keywords)}
        
        for k, v in attrs.items():
            lines.append(f"{k:<20} | {str(v):<20}")
        return "\n".join(lines)

