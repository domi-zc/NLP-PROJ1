import argparse
import lightning as L
import wandb

from src.data import ARTDataModule
from src.models import ARTModule
from src.train import WANDB_PROJECT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--wandb_run_id")
    args = parser.parse_args()

    dm = ARTDataModule()
    dm.setup()
    model = ARTModule.load_from_checkpoint(args.checkpoint, embeddings=dm.embeddings)

    trainer = L.Trainer(accelerator="auto", devices=1, logger=False)
    val = trainer.validate(model, datamodule=dm)[0]
    test = trainer.test(model, datamodule=dm)[0]
    print(f"val_acc={val['val_acc']:.4f}  test_acc={test['test_acc']:.4f}")

    if args.wandb_run_id:
        run = wandb.init(project=WANDB_PROJECT, id=args.wandb_run_id, resume="must")
        run.summary["test_acc"] = test["test_acc"]
        run.summary["test_loss"] = test["test_loss"]
        run.finish()


if __name__ == "__main__":
    main()
