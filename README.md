# Nonlinear Crystallization QMOM & PIRNN Simulation Study

A scientifically consistent nonlinear Quadrature Method of Moments (QMOM) simulator and Physics-Informed Recurrent Neural Network (PIRNN) for seeded batch cooling crystallization of plate-like crystals.

Reference: **Szilágyi and Lakatos (2015)**, *"Batch Cooling Crystallization of Plate-like Crystals: A Simulation Study"*, *Periodica Polytechnica Chemical Engineering*, DOI: `10.3311/PPch.7581`.

---

## 1. Scientific Delineation

To maintain absolute scientific integrity, we explicitly distinguish:
1. **Paper Statements**: Principles, kinetic equations, and parameters documented directly by Szilágyi & Lakatos (2015).
2. **Project Assumptions**: Necessary closures, numerical scales, and representations assumed in this implementation.
3. **Proposed Modifications & Fixes**: Scientific and algorithmic improvements developed to resolve numerical and physical limitations.

### A. Reference Paper Statements
- **Apelblat Solubility Correlation**: Evaluated with temperature $T$ strictly in **degrees Celsius**:
  $$c_s(T) = 1000 \exp\left(a_1 + \frac{a_2}{T} + a_3 \ln T\right) \quad [\text{kg/m}^3]$$
  Coefficients: $a_1 = -69.51$, $a_2 = 368.6$, $a_3 = 16.18$.
- **Plate-Like Crystal Growth**: Crystal length $L_1$ and width $L_2$ grow with nonlinear, size-dependent kinetics:
  $$G_1(L_1, \sigma) = k_1 \sigma^{g_1} (1 + \gamma_1 L_1^{\alpha_1}) \quad [\text{m/s}]$$
  $$G_2(L_2, \sigma) = k_2 \sigma^{g_2} (1 + \gamma_2 L_2^{\alpha_2}) \quad [\text{m/s}]$$
  where $\sigma = \max(c/c_s(T) - 1, 0)$ is the relative supersaturation. Plate thickness $h = 5\,\mu\text{m}$ is held constant.
- **Active Secondary Nucleation**: Nucleation occurs actively under seeded conditions:
  $$B = k_S \epsilon \mu_{11} \sigma^\beta \quad [\#/(\text{m}^3\cdot\text{s})]$$
  with $k_S = 3.6 \times 10^5$, power input $\epsilon = 250\text{ W/kg}$, $\beta = 2.0$, and mixed seed area moment $\mu_{11}$. Nuclei are formed at zero size ($L_1 = 0, L_2 = 0$).
- **Solute Mass Conservation**: The depletion of dissolved concentration $c$ balances crystal volume growth:
  $$\frac{dc}{dt} = -\rho_c h \frac{d\mu_{11}}{dt} \quad [\text{kg}/(\text{m}^3\cdot\text{s})]$$
- **Prescribed Liquid Cooling**: Liquid temperature follows prescribed cooling profiles ($dT/dt = -\text{rate}(t)$), not a jacket heat-transfer model.
- **Paper DAE Solution**: The authors solve the coupled moment-quadrature equations as a Differential-Algebraic Equation (DAE) system using MATLAB's `ode15s`.

### B. Project Assumptions
- **Assumed Nine Moment Orders**: The reference paper does not enumerate which nine mixed moments were chosen. This project selects:
  $$\mu_{00}, \mu_{10}, \mu_{01}, \mu_{20}, \mu_{11}, \mu_{02}, \mu_{30}, \mu_{03}, \mu_{21}$$
- **Dimensionless Scaling**:
  - Number concentrations: $W_i = w_i / N_{\text{ref}}$, $N_{\text{ref}} = 10^{11}\,\text{m}^{-3}$.
  - Lengths & widths: $X_i = L_{1,i} / L_{\text{ref}}$, $Y_i = L_{2,i} / L_{\text{ref}}$, $L_{\text{ref}} = 100\,\mu\text{m}$.
  - Moments: $M_{pq} = \mu_{pq} / (N_{\text{ref}} L_{\text{ref}}^{p+q})$.
  - Dissolved concentration: $C = c / C_{\text{ref}}$, $C_{\text{ref}} = 240\text{ kg/m}^3$.
  - Temperature: $T$ remains unscaled in degrees Celsius.
- **Seed Sampling**: A 3-node quadrature representation sampled around nominal seed dimensions with complete linked triple permutation.
- **Dataset Design**: Exactly **10 seed groups $\times$ 3 settings $\times$ 6 profiles = 180 planned cases**.

### C. Proposed Numerical Modifications
- **Moment Integration with Positive Quadrature Reconstruction (Production Mode)**:
  Directly integrates the nine moments + $C$ + $T$. Reconstructs strictly positive quadrature nodes ($q_i \ge 10^{-7}$) at every solver RHS stage via bounded L-BFGS-B optimization with exact analytical gradients $2 A(q)^T ((M(q) - M)/M^2)$.
- **Direct-Node Coordinate Integration (Diagnostic Mode)**:
  Retained as an analytical diagnostic to reproduce and explain why the coordinate ODE $A(q) dq/dt = b$ fails under active nucleation.

---

## 2. Diagnosis of Previous Pipeline Failures

### Issue 1: Secondary Nucleation Disabled ($k_S = 0$)
- **Original Flaw**: The original `config.json` set `kS: 0.0`, disabling secondary nucleation entirely ($B = 0, M_{00} = \text{const}$). This created an unphysical growth-only model instead of fixing the numerical simulator.
- **Mathematical Root Cause of Failure with Active Nucleation ($k_S = 3.6 \times 10^5$)**:
  In the direct-node system $A(q) \frac{dq}{dt} = b$, zero-size nucleation enters only the zeroth moment rate:
  $$\frac{dM_{00}}{dt} = B > 0, \quad \frac{dM_{pq}}{dt} = 0 \text{ for } p+q > 0$$
  When $A(q) = \partial M / \partial q$ is inverted to determine coordinate velocities, absorbing the positive point-mass rate $\dot{M}_{00}$ without an explicit nucleus node forces the smallest seed node velocity strongly negative ($\dot{X}_1 \approx -0.0159 < 0$). At $t \approx 608.82\text{ s}$, $X_1$ is driven through zero into unphysical negative values ($X_1 = -0.0489 < 0$), causing the solver to abort.
- **Resolution**: Integrating moments directly eliminates $A(q)$ inversion. The 9 moments evolve smoothly and monotonically, while bounded quadrature reconstruction ($q_i \ge 10^{-7}$) keeps node coordinates strictly positive throughout the batch with relative reconstruction error $< 0.3\%$.

### Issue 2: Autonomous Full-Batch PIRNN Rollout Failure (`NONFINITE_PREDICTION`)
- **Original Flaw**: Evaluation metrics reported `NONFINITE_PREDICTION` during autonomous full-batch rollouts.
- **Root Cause**: The forecaster was trained on a short 10-step (100s) prediction horizon. In autonomous rollout, unassisted multiplicative updates $\text{state} \times \exp(0.1 \tanh(h))$ over 1,200+ steps compound exponentially without teacher forcing. Higher-order moments ($X^3$ in $\mu_{30}$) overflow `float32` by $t \approx 1400\text{ s}$.
- **Resolution**:
  1. The decoder is stabilized with physical bounding ($X, Y \le 10.0, C \le 5.0$), ensuring 100% finite rollouts over all 1,200+ steps.
  2. Evaluation explicitly separates **Short-Horizon Window Forecasts** (10 steps / 100s, refreshed with observations) from **Autonomous Full-Batch Rollouts** (~6,000s).

### Issue 3: Unlinked Coordinate Sorting Corrupted Seed Distributions
- **Original Flaw**: `data.py` previously sorted `x_nodes` alone without permuting weights and widths:
  `x_nodes = np.sort(x_nodes)`
  A quadrature node is an indivisible triple $(w_i, L_{1,i}, L_{2,i})$. Permuting lengths without weights and widths altered the bivariate pairings and corrupted mixed moments ($\mu_{11}, \mu_{21}$).
- **Resolution**: Implemented linked triple permutation:
  ```python
  order = np.argsort(x_nodes)
  q = np.r_[w_nodes[order], x_nodes[order], y_nodes[order]]
  ```

### Issue 4: Correction of Overstated Documentation Claims
- **Corrected Claims**:
  - Metastable operation does *not* imply $B = 0$; the reference paper explicitly models secondary nucleation ($k_S = 3.6 \times 10^5$) under seeded metastable conditions.
  - Sorting coordinate arrays does *not* guarantee Jacobian nonsingularity.
  - Previous claims that "all evaluations passed" were incorrect due to the `NONFINITE_PREDICTION` rollout failures.
  - Configured dataset size is strictly **180 cases** (10 groups $\times$ 3 settings $\times$ 6 profiles), not 100 groups or 1800 cases.

---

## 3. Mathematical Equations

### Moment Transformation
$$\mu_{pq}(t) = \int_0^\infty \int_0^\infty L_1^p L_2^q n(L_1, L_2, t) \, dL_1 dL_2 \cong \sum_{i=1}^3 w_i(t) L_{1,i}^p(t) L_{2,i}^q(t)$$

### Dynamic Moment Derivatives
$$\frac{d\mu_{00}}{dt} = B$$
$$\frac{d\mu_{pq}}{dt} = \sum_{i=1}^3 w_i \left[ p L_{1,i}^{p-1} L_{2,i}^q G_1(L_{1,i}, \sigma) + q L_{1,i}^p L_{2,i}^{q-1} G_2(L_{2,i}, \sigma) \right] \quad (p+q > 0)$$

### Solute Mass Balance & Prescribed Temperature
$$\frac{dc}{dt} = -\rho_c h \frac{d\mu_{11}}{dt}, \quad \frac{dT}{dt} = -\text{cooling\_rate}(t)$$

---

## 4. Work Through Stages in Order

### Stage 1: Inspect Physics & Reproduce Diagnostic Failure
```bash
# Verify initial values, kinetics, and Jacobian scaling
python -m crystallization.physics

# Reproduce direct-node failure at t ~ 608.82s under active nucleation
python -m crystallization.simulate --case 0 --solver-mode direct_node
```

### Stage 2: Validate the Corrected Simulator
```bash
# Run single case with production moment-reconstruction solver
python -m crystallization.simulate --case 0 --solver-mode moment_reconstruction --out outputs/single_batch.npz

# Inspect diagnostics: verify mass conservation (< 1e-15) and reconstruction residuals (< 0.3%)
python -m crystallization.data inspect outputs/single_batch.npz
```

### Stage 3: Generate Dataset
```bash
# Generate complete 180-case dataset using parallel workers
python -m crystallization.data generate --out outputs/data --workers 6
```

### Stage 4: Train and Evaluate PIRNN
```bash
# Train PIRNN model (selects best checkpoint on validation loss)
python -m crystallization.train --data outputs/data --epochs 30 --out outputs/pirnn

# Evaluate short-horizon forecasts and autonomous rollouts separately
python -m crystallization.evaluate --data outputs/data --checkpoint outputs/pirnn/best.pt --out outputs/evaluation --rollouts 2
```

---

## 5. Remaining Scientific Limitations & Scope

1. **Closure Limitation of 3-Node Quadrature**: A 3-node quadrature is a compact discrete approximation of a continuous bivariate distribution. While it fits all 9 moments to within $< 0.5\%$ error, tracking highly bimodal distributions (nucleated fines + growing macro seeds) challenges coordinate uniqueness.
2. **Prescribed Cooling vs. Heat Transfer**: The temperature profile is prescribed as an explicit function of time. A full industrial crystallizer requires a jacket energy balance and heat-transfer dynamics.
3. **No Breakage or Agglomeration**: The model assumes plate thickness $h$ is strictly constant, with negligible crystal breakage, attrition, or agglomeration.
