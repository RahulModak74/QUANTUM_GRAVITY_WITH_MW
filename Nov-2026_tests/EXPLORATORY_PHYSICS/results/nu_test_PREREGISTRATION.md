# ν > 1 falsification test: pre-registration

Written 2026-11-19, **before any ν > 1 model was trained.**

## Prediction under test (March 2026 paper, Remark 1)

> Replacing the Student-t(ν = 0.8) prior with any distribution having ν > 1 should destroy the Planck-scale foam structure while preserving the GR sector.

## Design

- Same v5 data as run A: train on generator seed 1, test on seed 2 (30,000 rows each), with the same architecture, the same 150 epochs and all six priors.
- Only one thing changes: the degrees of freedom ν of the Student-t measurement prior (C6).
- Grid: ν ∈ {0.8, 1.5, 3.0} × training seed ∈ {0, 1}. The (0.8, seed 0) cell is the existing checkpoint `A_full_prior`, retrained here so every cell goes through the same script. Training seeds measure run-to-run noise.

## Measures (decoded test states)

| Sector | Measure |
|---|---|
| GR | Einstein trace residual (as in `results.json`); Hawking slope and R² |
| Planck/topology | (P1) accuracy of the decoded `topology` and `disc_flag` columns, thresholded at 0.5; (P2) mean absolute error of those two columns; (P3) wormhole recall of a 10-nearest-neighbour classifier on the encoder means, i.e. how well topology-changing states separate in latent space |

## Decision rule

The ν = 0.8 band is the range over its two seeds.

- **GR preserved:** for both ν values, every seed has an Einstein residual ≤ 1.5 × the worse ν = 0.8 seed.
- **Planck structure degraded:** for at least one ν > 1 value, both seeds are worse than the entire ν = 0.8 band on at least two of P1–P3.
- **Prediction supported** if GR is preserved and the Planck structure is degraded.
- **Prediction falsified** if the Planck structure is not degraded, or if GR is not preserved.

The March text also says ν > 1 gives a "0% Einstein residual". That is reported as a number, not used as the criterion.

One seed pair per ν; results are reported as they come out.
