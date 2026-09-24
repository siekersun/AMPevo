"""Fine-tune the original ESM2-650M with peptide masked-language modeling.

The dataset is supplied as the predefined UniProt train/validation/test split.

Example:
    python finetune_esm.py \
        --train-csv data/esm_train.csv \
        --val-csv data/esm_validation.csv \
        --test-csv data/esm_test.csv \
        --sequence-column Sequence \
        --output models/esm_finetuned.pth \
        --device cuda:0 --batch-size 2 --epochs 10
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model.pretrain import esm2_t33_650M_UR50D


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


class MLMCollater:
    """Apply standard 80/10/10 BERT masking to amino-acid tokens."""

    def __init__(self, alphabet, mask_probability=0.15):
        self.alphabet = alphabet
        self.converter = alphabet.get_batch_converter()
        self.mask_probability = mask_probability
        self.special_ids = {
            alphabet.padding_idx,
            alphabet.cls_idx,
            alphabet.eos_idx,
        }
        # Standard amino-acid token IDs; random replacements are drawn only
        # from these tokens, never from padding or other special symbols.
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.amino_acid_ids = torch.tensor(
            [alphabet.get_idx(amino_acid) for amino_acid in amino_acids],
            dtype=torch.long,
        )

    def __call__(self, sequences):
        data = [(f"sequence_{i}", sequence) for i, sequence in enumerate(sequences)]
        _, _, target = self.converter(data)
        valid = torch.ones_like(target, dtype=torch.bool)
        for token_id in self.special_ids:
            valid &= target.ne(token_id)

        selected = torch.rand(target.shape) < self.mask_probability
        selected &= valid

        # Ensure every sequence contributes at least one supervised residue.
        for row in range(target.size(0)):
            if not selected[row].any():
                positions = torch.where(valid[row])[0]
                if len(positions):
                    chosen = positions[torch.randint(len(positions), (1,))]
                    selected[row, chosen] = True

        source = target.clone()
        choice = torch.rand(target.shape)
        replace_with_mask = selected & (choice < 0.8)
        replace_randomly = selected & (choice >= 0.8) & (choice < 0.9)
        source[replace_with_mask] = self.alphabet.mask_idx

        random_indices = torch.randint(
            len(self.amino_acid_ids), target.shape
        )
        random_tokens = self.amino_acid_ids[random_indices]
        source[replace_randomly] = random_tokens[replace_randomly]

        # Unselected positions and the remaining 10% stay unchanged. Ignore
        # unselected positions in cross-entropy.
        labels = target.clone()
        labels[~selected] = -100
        return source.long(), labels.long(), selected


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ESM2-650M on peptide sequences")
    parser.add_argument("--train-csv", default="data/esm_train.csv")
    parser.add_argument("--val-csv", default="data/esm_validation.csv")
    parser.add_argument("--test-csv", default="data/esm_test.csv")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--output", default="models/esm_finetuned.pth")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--train-last-n-layers",
        type=int,
        default=1,
        help="Only fine-tune the last N transformer layers (1 is memory-efficient)",
    )
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_sequences(path, column, max_length):
    frame = pd.read_csv(path)
    if column not in frame.columns:
        raise KeyError(f"Column {column!r} not found; available columns: {list(frame.columns)}")
    sequences = frame[column].dropna().astype(str).str.strip().str.upper()
    sequences = sequences[(sequences.str.len() > 0) & (sequences.str.len() <= max_length)]
    if sequences.empty:
        raise ValueError("No usable sequences remain after filtering")
    return sequences.tolist()


def model_logits(model, tokens):
    # In this repository ESM2.forward returns [B, L, 1280] representations,
    # whereas lm_head maps those representations to amino-acid-token logits.
    return model.lm_head(model(tokens))


def configure_trainable_parameters(model, last_n_layers):
    """Freeze ESM and enable only the final transformer block(s) and output head."""
    if not 0 <= last_n_layers <= len(model.layers):
        raise ValueError(
            f"--train-last-n-layers must be between 0 and {len(model.layers)}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    if last_n_layers:
        for layer in model.layers[-last_n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    for parameter in model.emb_layer_norm_after.parameters():
        parameter.requires_grad = True

    # The LM-head output weight is tied to embed_tokens.weight. Keeping that
    # shared matrix frozen prevents gradients from starting at the input and
    # retaining activations through all 33 transformer layers. The smaller
    # dense, layer-norm and bias terms in the head can still be trained.
    for name, parameter in model.lm_head.named_parameters():
        if name != "weight":
            parameter.requires_grad = True

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Trainable parameters: {trainable:,}/{total:,} "
        f"({100.0 * trainable / total:.3f}%); "
        f"last transformer layers: {last_n_layers}"
    )
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def evaluate(model, loader, device, description="Evaluating"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.inference_mode():
        progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=False)
        for source, labels, selected in progress:
            source = source.to(device)
            labels = labels.to(device)
            selected = selected.to(device)
            with autocast(enabled=device.type == "cuda"):
                logits = model_logits(model, source)
                loss_sum = F.cross_entropy(
                    logits.transpose(1, 2), labels, ignore_index=-100, reduction="sum"
                )
            predictions = logits.argmax(dim=-1)
            count = int(selected.sum().item())
            total_loss += float(loss_sum.item())
            total_correct += int(((predictions == labels) & selected).sum().item())
            total_tokens += count
            progress.set_postfix(
                loss=f"{total_loss / max(total_tokens, 1):.4f}",
                accuracy=f"{total_correct / max(total_tokens, 1):.4f}",
            )
    return total_loss / total_tokens, total_correct / total_tokens


def save_weights(model, output, metadata):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Save the raw ESM state_dict so it can be loaded directly by the encoding
    # and comparison scripts created for this project.
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, output)
    with open(output.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("Batch size and gradient accumulation must be at least 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_sequences = load_sequences(
        args.train_csv, args.sequence_column, args.max_length
    )
    validation_sequences = load_sequences(
        args.val_csv, args.sequence_column, args.max_length
    )
    test_sequences = load_sequences(
        args.test_csv, args.sequence_column, args.max_length
    )
    train_dataset = SequenceDataset(train_sequences)
    validation_dataset = SequenceDataset(validation_sequences)
    test_dataset = SequenceDataset(test_sequences)

    print("Loading official ESM2-650M as the fine-tuning starting point...")
    model, alphabet = esm2_t33_650M_UR50D()
    device = torch.device(args.device)
    model.to(device)
    trainable_parameters = configure_trainable_parameters(
        model, args.train_last_n_layers
    )

    collater = MLMCollater(alphabet, args.mask_probability)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collater,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collater,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collater,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=device.type == "cuda")
    best_validation_loss = float("inf")

    print(
        f"Training sequences: {len(train_dataset)}; "
        f"validation sequences: {len(validation_dataset)}; "
        f"test sequences: {len(test_dataset)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_tokens = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{args.epochs}",
            dynamic_ncols=True,
        )
        for step, (source, labels, selected) in enumerate(progress, start=1):
            source = source.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            selected = selected.to(device, non_blocking=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model_logits(model, source)
                loss = F.cross_entropy(
                    logits.transpose(1, 2), labels, ignore_index=-100, reduction="mean"
                )
                scaled_loss = loss / args.gradient_accumulation

            scaler.scale(scaled_loss).backward()
            should_step = (
                step % args.gradient_accumulation == 0 or step == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            token_count = int(selected.sum().item())
            running_loss += float(loss.item()) * token_count
            running_tokens += token_count
            progress.set_postfix(
                loss=f"{running_loss / max(running_tokens, 1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_loss = running_loss / running_tokens
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device, description="Validation"
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs}: train_loss={train_loss:.6f}, "
            f"val_loss={validation_loss:.6f}, val_masked_accuracy={validation_accuracy:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            metadata = {
                "base_model": "esm2_t33_650M_UR50D",
                "train_dataset": args.train_csv,
                "validation_dataset": args.val_csv,
                "test_dataset": args.test_csv,
                "sequence_column": args.sequence_column,
                "epoch": epoch,
                "validation_loss": validation_loss,
                "validation_masked_accuracy": validation_accuracy,
                "arguments": vars(args),
            }
            save_weights(model, args.output, metadata)
            print(f"Saved new best fine-tuned ESM weights to {args.output}")

    # Test only once, using the checkpoint selected by validation loss.
    best_state = torch.load(args.output, map_location="cpu")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    test_loss, test_accuracy = evaluate(
        model, test_loader, device, description="Test"
    )
    print(
        f"Best-checkpoint test result: loss={test_loss:.6f}, "
        f"masked_accuracy={test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
