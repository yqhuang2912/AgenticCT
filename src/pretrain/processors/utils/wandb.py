import wandb
import os

def config_wandb(project_name, run_name):
    """
    Configure wandb with the provided arguments.
    """
    os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
    wandb.login(key="425bce26ce0da7d440ca0ace5f1dddc786d8498c", relogin=True)
    wandb.init(
        project=project_name,
        name=run_name,
    )