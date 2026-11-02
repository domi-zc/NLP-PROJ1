import argparse
import lightning as L

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from src.data import ARTDataModule
from src.models import ARTModule

WANDB_PROJECT = "nlp-proj1-final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture", choices=["embedding", "rnn"], default="embedding"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--classifier_hidden", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rnn_hidden", type=int, default=256)
    parser.add_argument("--tags", nargs="*", default=[])
    return parser.parse_args()


def run_name(args: argparse.Namespace) -> str:
    common = (
        f"lr{args.learning_rate:g}-do{args.dropout}-wd{args.weight_decay:g}"
        f"-mlp{args.classifier_hidden}-s{args.seed}"
    )
    if args.architecture == "embedding":
        return f"emb-{common}"
    return f"rnn-gru{args.rnn_hidden}-{common}"


def main():
    args = parse_args()
    L.seed_everything(args.seed)

    dm = ARTDataModule(batch_size=args.batch_size)
    dm.setup()

    model_kwargs = {
        "classifier_hidden": args.classifier_hidden,
        "dropout": args.dropout,
    }
    
    if args.architecture == "rnn":
        model_kwargs["rnn_hidden"] = args.rnn_hidden

    model = ARTModule(
        dm.embeddings,
        architecture=args.architecture,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        **model_kwargs,
    )

    name = run_name(args)
    logger = WandbLogger(
        project=WANDB_PROJECT,
        name=name,
        group=args.architecture,
        tags=[args.architecture, *args.tags],
        config=vars(args),
    )
    checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/{logger.experiment.id}-{name}",
        filename="best-{epoch}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_acc", mode="max", patience=args.patience
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint, early_stopping],
        log_every_n_steps=20,
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)

    logger.experiment.summary["best_val_acc"] = checkpoint.best_model_score.item()
    logger.experiment.summary["best_checkpoint"] = checkpoint.best_model_path
    logger.experiment.finish()


if __name__ == "__main__":
    main()
