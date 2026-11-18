**Think Manifolds, Not PDEs: our quantum gravity paper, revised and re-tested**

As we close out the work done in 2026, I went back to the Modak–Walawalkar (MW) quantum gravity paper we published in March and used the latest Claude Code to rework the model, the code and the claims. It did the job remarkably well.

**What changed**

The earlier version was open to a fair objection: that it was circular, simply reproducing physics already written into the training data. The answer lies in the difference between individual values and their joint distribution.

Salt tastes salty in any amount. Yet inside a curry, below a threshold, you don't taste it at all. That property belongs to the salt *together with* the curry; you can't find it by studying salt alone. A variational autoencoder learns exactly this: the joint distribution, not the individual columns.

This time we tested that directly. We kept every value in the dataset but shuffled which values appear together. The physics disappeared: the Einstein residual went from 1.3% to 60%, and the Hawking relation vanished. What the MW manifold holds is the joint structure itself.

**Three results, each with a control**

1. **One manifold holds GR and QM together.** From a single latent point, the Einstein trace relation holds to 1.3% (the earlier version reported 3.3%), and Hawking scaling T_H ∝ 1/M is recovered with slope −0.93.
2. **MW Bayesian priors make the joint fit faithful.** The same network without MW priors recovers the Hawking relation at R² 0.42; with them, R² 0.83.
3. **The manifold carries the joint, not the parts.** That's the shuffle test above.

**Why it matters**

Every major approach to holding general relativity and quantum mechanics in one framework changes something fundamental: extra dimensions (string theory), discretised spacetime (LQG, CDT), or a restricted geometry (AdS/CFT). The MW manifold needs none of these, and it is a single object.

**And it removes the coupling step that PDE-based methods require.** Coupling theories through their equations means a shared mesh, interface terms, and time steps spanning more than 55 orders of magnitude, from the Planck length to astrophysical scales. MW models the surface where valid states live instead. In practice, coupling theories becomes density estimation on one manifold.

Think manifolds, not PDEs.

**Try it, break it, improve it**

The revised paper (v2), all code, and every run are open in the `Nov-2026_tests` folder:
https://github.com/RahulModak74/QUANTUM_GRAVITY_WITH_MW/tree/master/Nov-2026_tests

Each run takes about 12 minutes on a laptop GPU. I'd welcome anyone interested to reproduce it, challenge it, and extend it, especially to real multi-physics data.

With Dr. Rahul Walawalkar. Built on PyTorch and Pyro.

#Bayesian #MachineLearning #QuantumGravity #Physics #VAE #ScientificComputing #OpenScience
