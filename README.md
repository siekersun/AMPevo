# AMPevo

AMPevo generates antimicrobial peptide sequences with an ESM-2 encoder and a ByteNet decoder. It supports generation from scratch, sequence inpainting, and motif scaffolding. Masked positions are predicted in parallel.

Run the commands below from the `AMPevo` directory. Generation requires a trained checkpoint at `checkpoints/ampevo.pth`; model weights are not included in this repository.

## Generate peptides

Generate 100 peptides of length 15:

```bash
python generate.py unconditional --checkpoint checkpoints/ampevo.pth --length 15 --num-sequences 100
```

Replace positions 1–4 and 12–15 of a peptide (positions start at 1; other residues stay fixed):

```bash
python generate.py inpainting --checkpoint checkpoints/ampevo.pth --sequence KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK --positions 1-4,12-15 --num-sequences 100
```

Generate 20-residue peptides while keeping `KLAKLAK` fixed from position 7:

```bash
python generate.py scaffolding --checkpoint checkpoints/ampevo.pth --length 20 --motif KLAKLAK --motif-start 7 --num-sequences 100
```

For inpainting or scaffolding, you can use `--template` instead. Use amino-acid letters for fixed positions and `X` for positions to generate, for example `XXXXKLAKLAXXXXX`. Results are saved as `generated_<mode>.csv` by default; use `--output path/to/results.csv` to choose another file. Use `--temperature` or `--top-k` to adjust sampling.
