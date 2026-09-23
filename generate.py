"""Generate peptide sequences with the three AMPevo generation modes."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from esmdiff.model import ByteNetLMTime
from esmdiff.pretrain import esm2_t33_650M_UR50D


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MASK_TEXT = "<mask>"


def parse_positions(text, length):
    """Parse 1-based positions such as ``1-4,8,12-15``."""
    positions = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            if start > end:
                raise ValueError(f"Invalid position range: {part}")
            positions.update(range(start - 1, end))
        else:
            positions.add(int(part) - 1)
    if not positions or min(positions) < 0 or max(positions) >= length:
        raise ValueError(f"Positions must be within 1-{length}")
    return positions


def normalize_template(template):
    """Convert X, *, and _ placeholders to ESM mask tokens."""
    template = template.strip().upper().replace(" ", "")
    template = template.replace("<MASK>", MASK_TEXT)
    for placeholder in ("X", "*", "_"):
        template = template.replace(placeholder, MASK_TEXT)
    probe = template.replace(MASK_TEXT, "")
    invalid = sorted(set(probe) - set(AMINO_ACIDS))
    if invalid:
        raise ValueError(f"Invalid template character(s): {', '.join(invalid)}")
    if MASK_TEXT not in template:
        raise ValueError("The template must contain at least one X, *, _, or <mask>")
    return template


def make_generation_input(args):
    """Return (masked template, original sequence or None)."""
    if args.mode == "unconditional":
        if not args.length or args.length < 1:
            raise ValueError("Unconditional mode requires --length")
        return MASK_TEXT * args.length, None

    if args.mode == "inpainting":
        if args.template:
            return normalize_template(args.template), None
        if not args.sequence or not args.positions:
            raise ValueError(
                "Inpainting requires --sequence and --positions, or --template"
            )
        sequence = args.sequence.strip().upper()
        invalid = sorted(set(sequence) - set(AMINO_ACIDS))
        if invalid:
            raise ValueError(f"Invalid amino acid(s): {', '.join(invalid)}")
        positions = parse_positions(args.positions, len(sequence))
        template = "".join(
            MASK_TEXT if index in positions else residue
            for index, residue in enumerate(sequence)
        )
        return template, sequence

    if args.template:
        return normalize_template(args.template), None
    if not args.length or not args.motif or not args.motif_start:
        raise ValueError(
            "Scaffolding requires --template, or --length, --motif, and --motif-start"
        )
    motif = args.motif.strip().upper()
    invalid = sorted(set(motif) - set(AMINO_ACIDS))
    start = args.motif_start - 1
    if invalid:
        raise ValueError(f"Invalid motif amino acid(s): {', '.join(invalid)}")
    if start < 0 or start + len(motif) > args.length:
        raise ValueError("The motif does not fit at --motif-start within --length")
    residues = [MASK_TEXT] * args.length
    residues[start : start + len(motif)] = motif
    return "".join(residues), None


def load_model(checkpoint_path, device):
    esm_model, alphabet = esm2_t33_650M_UR50D()
    model = ByteNetLMTime(
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, alphabet


def sample_amino_acids(logits, aa_token_ids, temperature, top_k):
    """Sample canonical amino acids from logits at all sequence positions."""
    aa_logits = logits[..., aa_token_ids] / temperature
    if top_k and top_k < len(aa_token_ids):
        values, indices = torch.topk(aa_logits, top_k, dim=-1)
        probabilities = torch.softmax(values, dim=-1)
        choices = torch.multinomial(probabilities.reshape(-1, top_k), 1)
        choices = choices.reshape(values.shape[:-1])
        aa_indices = torch.gather(indices, -1, choices.unsqueeze(-1)).squeeze(-1)
    else:
        probabilities = torch.softmax(aa_logits, dim=-1)
        aa_indices = torch.multinomial(
            probabilities.reshape(-1, len(aa_token_ids)), 1
        ).reshape(aa_logits.shape[:-1])
    return aa_token_ids[aa_indices]


def decode_tokens(tokens, alphabet):
    return "".join(alphabet.get_tok(int(token)) for token in tokens)


def template_length(template):
    return len(template.replace(MASK_TEXT, "X"))


@torch.inference_mode()
def generate_candidates(
    model,
    alphabet,
    template,
    batch_size,
    device,
    temperature,
    top_k,
):
    # Generation must preserve the caller's masks. The training collater is not
    # used here because it applies a new random mask to every input sequence.
    converter = alphabet.get_batch_converter()
    batch = [(f"sample_{i}", template) for i in range(batch_size)]
    _, _, source = converter(batch)
    source = source.to(device)
    mutable = source.eq(alphabet.mask_idx)
    input_mask = source.ne(alphabet.padding_idx).unsqueeze(-1).float()

    logits = model(source, input_mask=input_mask)
    aa_token_ids = torch.tensor(
        [alphabet.get_idx(amino_acid) for amino_acid in AMINO_ACIDS],
        device=device,
    )
    sampled = sample_amino_acids(logits, aa_token_ids, temperature, top_k)
    result = source.clone()
    result[mutable] = sampled[mutable]

    sequence_length = template_length(template)
    return [
        decode_tokens(row[1 : sequence_length + 1], alphabet) for row in result
    ]


def mutable_indices(template):
    """Return residue indices occupied by mask tokens in a template."""
    indices = []
    residue_index = 0
    cursor = 0
    while cursor < len(template):
        if template.startswith(MASK_TEXT, cursor):
            indices.append(residue_index)
            cursor += len(MASK_TEXT)
        else:
            cursor += 1
        residue_index += 1
    return set(indices)


def generate_unique(
    model,
    alphabet,
    template,
    original,
    count,
    batch_size,
    device,
    temperature,
    top_k,
    min_changes,
    max_rounds,
):
    sequences = set()
    positions = mutable_indices(template)

    for _ in range(max_rounds):
        needed = count - len(sequences)
        if needed <= 0:
            break
        candidates = generate_candidates(
            model,
            alphabet,
            template,
            min(batch_size, max(needed * 2, 1)),
            device,
            temperature,
            top_k,
        )
        for sequence in candidates:
            if original is not None and min_changes:
                changes = sum(sequence[index] != original[index] for index in positions)
                if changes < min_changes:
                    continue
            sequences.add(sequence)
            if len(sequences) == count:
                break

    if len(sequences) < count:
        raise RuntimeError(
            f"Only generated {len(sequences)} unique sequences after {max_rounds} "
            "rounds. Increase --temperature or --top-k, or reduce --num-sequences."
        )
    return sorted(sequences)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate peptides using AMPevo's three generation modes."
    )
    parser.add_argument(
        "mode", choices=("unconditional", "inpainting", "scaffolding")
    )
    parser.add_argument("--checkpoint", required=True, help="AMPevo checkpoint")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--num-sequences", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--length", type=int, help="Generated peptide length")
    parser.add_argument("--template", help="Fixed residues plus X/*/_ mask positions")
    parser.add_argument("--sequence", help="Original sequence for inpainting")
    parser.add_argument("--positions", help="1-based inpainting positions, e.g. 1-4,12")
    parser.add_argument("--motif", help="Motif retained in scaffolding mode")
    parser.add_argument("--motif-start", type=int, help="1-based motif start position")
    parser.add_argument(
        "--min-changes",
        type=int,
        default=None,
        help="Minimum changes at inpainted positions (default: 1 for inpainting)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.num_sequences < 1 or args.batch_size < 1:
        raise ValueError("--num-sequences and --batch-size must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero")
    if args.top_k < 0 or args.top_k > len(AMINO_ACIDS):
        raise ValueError(f"--top-k must be between 0 and {len(AMINO_ACIDS)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    template, original = make_generation_input(args)
    min_changes = args.min_changes
    if min_changes is None:
        min_changes = 1 if args.mode == "inpainting" and original else 0
    if min_changes < 0:
        raise ValueError("--min-changes cannot be negative")
    if original is not None and min_changes > len(mutable_indices(template)):
        raise ValueError("--min-changes exceeds the number of inpainted positions")

    model, alphabet = load_model(args.checkpoint, args.device)
    sequences = generate_unique(
        model,
        alphabet,
        template,
        original,
        args.num_sequences,
        args.batch_size,
        args.device,
        args.temperature,
        args.top_k,
        min_changes,
        args.max_rounds,
    )

    output_path = Path(args.output or f"generated_{args.mode}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["peptide"])
        writer.writerows((sequence,) for sequence in sequences)
    print(f"Saved {len(sequences)} unique sequences to {output_path}")


if __name__ == "__main__":
    main()
