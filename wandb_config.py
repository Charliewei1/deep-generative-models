import wandb

def init_wandb(config):
    wandb.init(
        project="deep-generative-models",
        config=config
    )

def log_metrics(metrics):
    wandb.log(metrics)