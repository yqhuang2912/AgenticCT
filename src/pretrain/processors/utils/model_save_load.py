import os
import logging
import json
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def save_model(model, config, save_path):
    """
    Save the model to the specified path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    with open(save_path.replace('.pth', '.json'), 'w') as f:
        json.dump(config, f)
    logger.debug(f"Model saved to {save_path}")


def load_model(Model: nn.Module, save_path):
    """
    Load the model from the specified path.
    """
    config_path = save_path.replace('.pth', '.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Model(**config)
    model.load_state_dict(torch.load(save_path))
    logger.debug(f"Model loaded from {save_path}")
    return model


if __name__ == "__main__":
    # Example usage
    class Model(nn.Module):
        def __init__(self, in_channels=3, out_channels=1):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = Model()
    save_path = "./model.pth"
    config = {"in_channels": 3, "out_channels": 1}
    
    save_model(model, config, save_path)
    loaded_model = load_model(Model, save_path)
    print(loaded_model)  # This will print the model architecture