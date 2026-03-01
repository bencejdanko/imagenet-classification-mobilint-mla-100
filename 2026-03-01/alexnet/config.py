class Config:
    IMAGE_ROOT = 'imagenet_train20a/imagenet_train20a'
    TRAIN_LIST = 'imagenet_train20.txt'
    VAL_IMAGE_ROOT = 'imagenet_val20/imagenet_val20'
    VAL_LIST = 'imagenet_val20.txt'
    BATCH_SIZE = 512
    NUM_CLASSES = 20
    INPUT_SHAPE = (240, 240)
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    
    # WandB and Checkpointing
    WANDB_PROJECT = 'alexnet-imagenet-20'
    WANDB_ENTITY = 'bdanko' # Set this to your wandb username if needed
    CHECKPOINT_DIR = 'checkpoints'
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

