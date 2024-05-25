# Lc0 interpretability

## Setup

```
pip install -e .
```

Download the files from (https://figshare.com/s/adc80845c00b67c8fce5) into your
working directory. (`lc0.onnx` and `interesting_puzzles.pkl` are required for most
experiments, `lc0-random.onnx` is only required for the probing baseline,
and `LD2.onnx` and `unfiltered_puzzles.pkl` are only required if you want to explore
other ways to filter or corrupt puzzles).

## Reproducing results

If you want to regenerate the puzzle dataset, follow these steps:
- Download `lichess_db_puzzle.csv.zst` from https://database.lichess.org/#puzzles.
  Then decompress using `zstd -d lichess_db_puzzle.csv.zst`.
- Run `scripts/make_puzzles.py --generate` to generate the puzzle dataset.
- Run `scripts/make_corruptions.py` to add corruptions to puzzles.

Alternatively, you can use the `interesting_puzzles.pkl` file we provide.

Then compute the raw results using the other scripts:
- For probing results, run `scripts/probing.py --main --random_model --n_seeds 5`
  (leave out `--random_model` if you don't want to run the baseline with probes trained
  on a randomly initialized model). You can use more or fewer seeds of course.
  Note that this requires around 70GB or RAM to store activations. You can
  alternatively store those activations on disk (see the source code of the script)
  or reduce the number of puzzles that are used with `--n_puzzles`.
- To get residual stream and attention head activation patching results, run
  `scripts/run_global_patching.py --residual_stream --attention`
- To reproduce the L12H12 results, run `scripts/run_single_head.py --main`. You can also
  study other heads with `--layer` and `--head`, and do additional ablations with
  `--squarewise` and `--single_weight`, as well as cache the attention pattern with
  `--attention_pattern`. None of these are necessary to reproduce the paper results.
- For piece movement head results, run `scripts/piece_movement_heads.py`.

Results will be stored in `results/`. Each script takes an `--n_puzzles` argument which
you can use to limit the number of puzzles to run on. You can often also specify
`--batch_size` and other arguments (in particular, you might want to set `--device`,
which defaults to `cuda`). Each script should run in at most a few hours on a
fast GPU like an A100.

Then use the `other_figures.ipynb` notebook to create most of the figures and other
analyses based on these results. `act_patching.ipynb`, `puzzle_example.ipynb`, and
`figure_1.ipynb` create Figures 1-3 with their illustrations.

Note that the notebooks assume the working directory is the root directory of this
repository (or whereever else the `lc0.onnx`, `interesting_puzzles.pkl`, and
`results/*` files are stored). You can set this in VS Code with
```json
{
    "jupyter.notebookFileRoot": "/path/to/dir"
}
```

## Using the codebase for follow-up work
We recommend you start by taking a look at `notebooks/demo.ipynb`, which shows how to
use some of the features of this codebase. From there, you can start exploring how the
scripts and the main `leela_interp` package work.

The `leela_interp` package has two subpackages: `leela_interp.core` and
`leela_interp.tools`. The `core` package contains infrastructure and general-purpose
utilities that will likely be useful whatever you do. The `tools` package is much more
optional---it contains code that we used or our experiments (e.g. probing),
but it might not suit your needs.

## Known issues
We've observed `NaN` outputs for Leela on MPS sometimes (but never on CPU or CUDA).