"""Train ByteNet/OADM on cluster-separated AMP data with a frozen adapted ESM."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model.losses import OAMaskedCrossEntropyLoss
from model.network import ByteNetLMTime
from model.pretrain import ESM2
from sequence_models.utils import warmup


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


class PeptideDataset(Dataset):
    def __init__(self, csv_path, sequence_column="sequence"):
        frame = pd.read_csv(csv_path)
        if sequence_column not in frame.columns:
            raise KeyError(
                f"Column {sequence_column!r} not found in {csv_path}; "
                f"available columns: {list(frame.columns)}"
            )
        sequences = frame[sequence_column].dropna().astype(str).str.strip().str.upper()
        valid = sequences[
            (sequences.str.len() > 0)
            & sequences.map(lambda sequence: set(sequence) <= CANONICAL_AA)
        ]
        if valid.empty:
            raise ValueError(f"No valid peptide sequences found in {csv_path}")
        self.sequences = valid.tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


class OAMaskCollater:
    """Mask 1..L residues; evaluation masks are deterministic per sequence."""

    def __init__(self, alphabet, deterministic=False, seed=42):
        self.alphabet = alphabet
        self.converter = alphabet.get_batch_converter()
        self.deterministic = deterministic
        self.seed = seed

    def _generator(self, sequence):
        if not self.deterministic:
            return None
        digest = hashlib.sha256(f"{self.seed}:{sequence}".encode()).digest()
        value = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        return torch.Generator().manual_seed(value)

    def __call__(self, sequences):
        batch = [(f"peptide_{i}", sequence) for i, sequence in enumerate(sequences)]
        _, _, target = self.converter(batch)
        source = target.clone()
        mask = torch.zeros_like(target, dtype=torch.bool)
        timesteps = []
        for row, sequence in enumerate(sequences):
            generator = self._generator(sequence)
            length = len(sequence)
            count = int(torch.randint(1, length + 1, (1,), generator=generator).item())
            positions = torch.randperm(length, generator=generator)[:count] + 1
            source[row, positions] = self.alphabet.mask_idx
            mask[row, positions] = True
            timesteps.append(count)
        return source.long(), torch.tensor(timesteps), target.long(), mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ByteNet/OADM with frozen short-peptide-adapted ESM"
    )
    parser.add_argument("--esm-weights", default="checkpoints/esm_finetuned.pth")
    parser.add_argument("--train-csv", default="data/amp_train.csv")
    parser.add_argument("--val-csv", default="data/amp_validation.csv")
    parser.add_argument("--test-csv", default="data/amp_test.csv")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--output", default="checkpoints/ampevo.pth")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_finetuned_esm(weight_path):
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=33,
        embed_dim=1280,
        attention_heads=20,
        alphabet=alphabet,
        token_dropout=True,
    )
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Adapted ESM checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model, alphabet


def build_model(esm_model, alphabet):
    return ByteNetLMTime(
        n_tokens=len(alphabet),
        d_model=esm_model.embed_dim,
        n_layers=30,
        kernel_size=5,
        dilation_cycle=128,
        esm_model=esm_model,
        causal=False,
        dropout=0.0,
        slim=True,
        activation="relu",
    )


def assert_disjoint_splits(train_dataset, validation_dataset, test_dataset):
    train = set(train_dataset.sequences)
    validation = set(validation_dataset.sequences)
    test = set(test_dataset.sequences)
    overlaps = {
        "train/validation": len(train & validation),
        "train/test": len(train & test),
        "validation/test": len(validation & test),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Exact sequence overlap across AMP splits: {overlaps}")


def run_epoch(
    model, loader, loss_function, padding_idx, device, description,
    optimizer=None, scaler=None, scheduler=None, gradient_accumulation=1,
):
    training = optimizer is not None
    if training:
        model.train()
        model.esm.eval()  # frozen encoder must remain deterministic
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    total_nll = 0.0
    total_correct = 0
    total_masked = 0
    progress = tqdm(loader, desc=description, dynamic_ncols=True)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for step, (source, timesteps, target, mask) in enumerate(progress, start=1):
            source = source.to(device, non_blocking=True)
            timesteps = timesteps.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            input_mask = source.ne(padding_idx).float()

            with autocast(enabled=device.type == "cuda"):
                logits = model(source, input_mask=input_mask.unsqueeze(-1))
                weighted_loss, nll = loss_function(
                    logits, target, mask, timesteps, input_mask
                )
                optimization_loss = weighted_loss / source.size(0)

            if training:
                scaler.scale(optimization_loss / gradient_accumulation).backward()
                should_step = step % gradient_accumulation == 0 or step == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            masked_count = int(mask.sum().item())
            correct = int(((logits.argmax(dim=-1) == target) & mask).sum().item())
            total_nll += float(nll.item())
            total_correct += correct
            total_masked += masked_count
            progress.set_postfix(
                nll=f"{total_nll / max(total_masked, 1):.4f}",
                accuracy=f"{total_correct / max(total_masked, 1):.4f}",
            )
    return total_nll / total_masked, total_correct / total_masked


def save_checkpoint(path, model, optimizer, scheduler, scaler, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if min(args.batch_size, args.validation_batch_size, args.gradient_accumulation) < 1:
        raise SystemExit("Batch sizes and gradient accumulation must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dataset = PeptideDataset(args.train_csv, args.sequence_column)
    validation_dataset = PeptideDataset(args.val_csv, args.sequence_column)
    test_dataset = PeptideDataset(args.test_csv, args.sequence_column)
    assert_disjoint_splits(train_dataset, validation_dataset, test_dataset)
    print(
        f"AMP data: train={len(train_dataset)}, validation={len(validation_dataset)}, "
        f"test={len(test_dataset)}"
    )

    print(f"Loading adapted ESM weights from {args.esm_weights}")
    esm_model, alphabet = load_finetuned_esm(args.esm_weights)
    model = build_model(esm_model, alphabet)
    device = torch.device(args.device)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(
        f"Trainable ByteNet/decoder parameters: {sum(p.numel() for p in trainable):,}; "
        "ESM encoder is frozen"
    )

    train_collater = OAMaskCollater(alphabet, deterministic=False, seed=args.seed)
    evaluation_collater = OAMaskCollater(alphabet, deterministic=True, seed=args.seed)
    common = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_collater, **common,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.validation_batch_size, shuffle=False,
        collate_fn=evaluation_collater, **common,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.validation_batch_size, shuffle=False,
        collate_fn=evaluation_collater, **common,
    )

    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, warmup(args.warmup_steps), verbose=False)
    scaler = GradScaler(enabled=device.type == "cuda")
    loss_function = OAMaskedCrossEntropyLoss(reweight=True)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, alphabet.padding_idx, device,
            f"Train {epoch:03d}/{args.epochs}", optimizer, scaler, scheduler,
            args.gradient_accumulation,
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, loss_function, alphabet.padding_idx,
            device, "Validation",
        )
        history.append(
            {"epoch": epoch, "train_nll": train_loss,
             "train_accuracy": train_accuracy, "validation_nll": validation_loss,
             "validation_accuracy": validation_accuracy}
        )
        print(
            f"Epoch {epoch:03d}: train_nll={train_loss:.6f}, "
            f"train_accuracy={train_accuracy:.4f}, val_nll={validation_loss:.6f}, "
            f"val_accuracy={validation_accuracy:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            metadata = {
                "epoch": epoch, "best_validation_nll": validation_loss,
                "validation_accuracy": validation_accuracy,
                "esm_weights": args.esm_weights, "esm_frozen": True,
                "train_csv": args.train_csv, "validation_csv": args.val_csv,
                "test_csv": args.test_csv,
                "architecture": {"esm": "esm2_t33_650M_UR50D",
                                 "bytenet_layers": 30, "kernel_size": 5, "r": 128},
                "arguments": vars(args),
            }
            save_checkpoint(args.output, model, optimizer, scheduler, scaler, metadata)
            print(f"Saved new best decoder checkpoint to {args.output}")
        else:
            epochs_without_improvement += 1

        history_path = Path(args.output).with_suffix(".history.csv")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if (args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience):
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    checkpoint = torch.load(args.output, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    test_loss, test_accuracy = run_epoch(
        model, test_loader, loss_function, alphabet.padding_idx, device, "Final test"
    )
    print(
        f"Best epoch {best_epoch}: test_nll={test_loss:.6f}, "
        f"test_accuracy={test_accuracy:.4f}"
    )
    with Path(args.output).with_suffix(".test.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"best_epoch": best_epoch, "test_nll": test_loss,
             "test_accuracy": test_accuracy}, handle, indent=2,
        )


if __name__ == "__main__":
    main()
