# QUANTUM_GRAVITY_WITH_MW

**One physics-prior Bayesian manifold that holds general relativity and quantum mechanics together**, built with the Modak–Walawalkar (MW) framework.

> **Think manifolds, not PDEs.**

---

## Start here: `Nov-2026_tests/` (current)

The current code, results and paper are in **[`Nov-2026_tests/`](Nov-2026_tests/)**.

- **Paper (v2, Nov 2026):** [`Nov-2026_tests/paper/mw_joint_manifold_gr_qm_nov2026.pdf`](Nov-2026_tests/paper/mw_joint_manifold_gr_qm_nov2026.pdf)
- **Code:** [`Nov-2026_tests/code/`](Nov-2026_tests/code/)
- **Results:** [`Nov-2026_tests/results/`](Nov-2026_tests/results/)

### Three results

| Result | Evidence (decoded test states) |
|---|---|
| **1. One VAE holds GR + QM jointly** | Einstein trace residual **1.3%**; Hawking T_H ∝ M^−0.93 (R² 0.83), from a single latent point |
| **2. MW priors improve joint fidelity** | Hawking R² **0.83 with MW priors vs 0.42** for the same VAE without them |
| **3. The manifold carries the joint, not the columns** | Shuffling columns within each regime (same values, broken rows): Einstein 1.3% → **59.6%**, Hawking R² → 0.19 |

![Nov-2026 results](Nov-2026_tests/results/fig_v5.png)

### Structural position

| Approach | Extra dim. | Discretise | AdS only | Single object |
|---|---|---|---|---|
| String theory | Yes | No | Partial | No |
| LQG | No | Yes | No | No |
| CDT | No | Yes | No | No |
| AdS/CFT | Yes | No | Yes | No |
| **MW (this work)** | **No** | **No** | **No** | **Yes** |

The other approaches are full theories of quantum gravity. MW is compared on structure: one learned manifold that holds GR and QM content together. **MW removes the coupling step that PDE-based methods require**, so coupling theories becomes joint density estimation on one manifold.

### Run it

```bash
pip install torch pyro-ppl numpy matplotlib
cd Nov-2026_tests/code
python3 qg_v5_experiments.py   # all runs -> ../results/results.json (~12 min per run on an RTX 3050)
python3 qg_v5_figures.py       # figure -> ../results/fig_v5.png and ../paper/fig_v5.pdf
```

Details are in [`Nov-2026_tests/README.md`](Nov-2026_tests/README.md).

---

## Repository map

| Path | What it is | Status |
|---|---|---|
| [`Nov-2026_tests/`](Nov-2026_tests/) | Improved code (v5), controlled runs, paper v2, LinkedIn post | **Current** |
| `qg_toy_data_generator_v4.py`, `qg_toy_vae_trainer_v4.py` | March 2026 data generator and trainer behind the first paper | Archive |
| `qg_toy_data_generator.py`, `qg_toy_vae_trainer.py`, `qg_toy_visualizer.py` | Earlier versions and visualiser | Archive |
| `qg_manifold_visualization_clean.png` | March 2026 figure | Archive |
| `create_mercury_orbital_data.py` | Mercury orbital data generator | Archive |
| [`THEORY/`](THEORY/) | March 2026 paper (`mw_quantum_gravity.pdf`), manifold and PDE extraction tools, theory notes | Archive |
| [`PDF_PAPERS_THEORY/`](PDF_PAPERS_THEORY/) | MW theory documents: field equations, QM as geometry, formal theory, differential-geometry primer | Theory |
| `MW_Framework_Quantum_Vision.pdf` | MW framework vision document | Theory |
| `STATUS_OF_QR_GM_DATA.md` | March 2026 response to reviewer objections | Archive |
| `LINKEDIN_ARTICLE.md` | March 2026 article | Archive |
| [`README_MARCH_2026.md`](README_MARCH_2026.md) | Full previous README | Archive |

The archive files are kept unchanged for reference and reproducibility. Paper v2 in `Nov-2026_tests/` supersedes the March 2026 preprint (Zenodo [10.5281/zenodo.19304813](https://zenodo.org/records/19304813)).

### What improved in v5

- **Data:** every row is one draw from one joint distribution, so each physical parameter is sampled once and all columns are derived from it.
- **Priors:** columns are read by name, so each physics prior acts on the intended column.
- **Controls:** each run is compared against a plain VAE and a shuffled-rows control.

---

## Citation

```bibtex
@software{modak2026quantum_gravity_mw_v2,
  author = {Modak, Rahul and Walawalkar, Rahul},
  title  = {One Physics-Prior Bayesian Manifold Jointly Encodes General-Relativistic
            and Quantum Constraints: Controlled Tests of the Modak--Walawalkar Framework},
  year   = {2026},
  month  = {11},
  url    = {https://github.com/RahulModak74/QUANTUM_GRAVITY_WITH_MW/tree/master/Nov-2026_tests},
  note   = {Paper v2; supersedes doi:10.5281/zenodo.19304813}
}
```

## Contact

**Rahul Modak**: rahul.modak@bayesiananalytics.in
Bayesian Cybersecurity Pvt Ltd, Mumbai/Thane, India

**Dr. Rahul Walawalkar**
Carnegie Mellon University; NETRA, Pune; Caret Capital

[MW Framework](https://github.com/RahulModak74/mw-framework)

## License

MIT, open for academic research. Commercial licensing for battery analytics and RF cybersecurity is separate.

---

Built on [PyTorch](https://pytorch.org) and [Pyro](https://pyro.ai). Code implementation was assisted by AI coding tools (Claude, Qwen, DeepSeek); all modelling choices and claims are the authors'.

**Think manifolds, not PDEs.**
