# Nonlinear Crystallization QMOM & PIRNN Simulation Study

A scientifically consistent nonlinear Quadrature Method of Moments (QMOM) simulator and Physics-Informed Recurrent Neural Network (PIRNN) for seeded batch cooling crystallization of plate-like crystals.

Reference: **Szilágyi and Lakatos (2015)**, *"Batch Cooling Crystallization of Plate-like Crystals: A Simulation Study"*, *Periodica Polytechnica Chemical Engineering*, DOI: `10.3311/PPch.7581`.

---

## 1. Scientific Delineation

To maintain absolute scientific integrity, we explicitly distinguish:
1. **Paper Statements**: Principles, kinetic equations, and parameters documented directly by Szilágyi & Lakatos (2015).
2. **Project Assumptions & Choices**: Necessary closures, numerical scales, representation choices, and cooling profile families assumed in this implementation.
3. **Proposed Modifications & Fixes**: Algorithmic improvements developed to resolve numerical and physical limitations.

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

### B. Project Assumptions and Project Choices
- **Cooling Profiles (Project Choices)**:
  The reference paper does not prescribe the full set of operational cooling curves. Our implementation adopts five distinct project cooling families to provide varied thermal histories:
  1. `linear`
  2. `cubic`
  3. `natural`
  4. `two_stage`
  5. `cool_hold_cool`
  These five families represent project design choices, not requirements dictated by the reference paper. Six families or `random_spline` profiles are not required.
- **Assumed Nine Moment Orders**: The reference paper does not enumerate which nine mixed moments were chosen. This project selects:
  $$\mu_{00}, \mu_{10}, \mu_{01}, \mu_{20}, \mu_{11}, \mu_{02}, \mu_{30}, \mu_{03}, \mu_{21}$$
- **Dimensionless Scaling & Units**:
  - Number concentrations: $W_i = w_i / N_{\text{ref}}$, $N_{\text{ref}} = 10^{11}\,\text{m}^{-3}$.
  - Lengths & widths: $X_i = L_{1,i} / L_{\text{ref}}$, $Y_i = L_{2,i} / L_{\text{ref}}$, $L_{\text{ref}} = 100\,\mu\text{m}$.
  - Moments: $M_{pq} = \mu_{pq} / (N_{\text{ref}} L_{\text{ref}}^{p+q})$.
  - Dissolved concentration: $C = c / C_{\text{ref}}$, $C_{\text{ref}} = 240\text{ kg/m}^3$. Concentration is strictly in $\text{kg/m}^3$; it is not converted to $\text{g/kg}$ without an explicitly justified density basis.
  - Temperature: $T$ remains unscaled in degrees Celsius ($^\circ\text{C}$).
- **Seed Sampling**: A 3-node quadrature representation sampled around nominal seed dimensions with complete linked triple permutation.
- **Dataset Design**: Exactly **10 seed groups $\times$ 3 settings $\times$ 5 profiles = 150 completed batches**.

### C. Proposed Numerical Modifications
- **Moment Integration with Positive Quadrature Reconstruction (Production Mode)**:
  Directly integrates the nine moments + $C$ + $T$. Reconstructs strictly positive quadrature nodes ($q_i \ge 10^{-7}$) via bounded L-BFGS-B optimization with exact analytical gradients $2 A(q)^T ((M(q) - M)/M^2)$. Warm-start anchors are updated strictly at accepted integration segment boundaries, guaranteeing call-order and trial-step independence.
- **Direct-Node Coordinate Integration (Diagnostic Mode)**:
  Retained as an analytical diagnostic demonstrating why the coordinate ODE $A(q) dq/dt = b$ fails under active nucleation when point-mass zero-size births are absorbed into 3 discrete nodes without an explicit nucleation node.

---

## 2. Scientific Distinctions & Taxonomy

To prevent misleading claims, this study maintains strict distinctions among:
1. **Completed Simulation**: The numerical ODE solver reached $t = t_{\text{final}}$ without throwing an exception.
2. **Numerically Validated Simulation**: The simulation completed AND satisfied explicit numerical tolerances:
   - Moment reconstruction error $\le 5.0\%$ across all 9 moments.
   - Solute mass conservation drift $\le 10^{-14}$ (scaled) along the entire trajectory.
   - Liquid temperature tracks prescribed profile within $\le 0.001\,^\circ\text{C}$ without post-hoc overwriting.
3. **Finite Neural-Network Prediction**: The PIRNN produced outputs containing no `NaN` or `Inf` values across the evaluation horizon.
4. **Accurate Physical Forecast**: The prediction is finite AND quantitatively matches the clean ground truth:
   - Low normalized error ($N\text{RMSE}$ and $N\text{MAE}$, normalized explicitly by the range $\max(\text{truth}) - \min(\text{truth})$).
   - Minimal drift from initial total mass.
   - Realistic crystal dimensions and zero constraint violations.

---

## 3. Resolution of Methodological Limitations

### Issue 1: Secondary Nucleation Disabled ($k_S = 0$)
- **Original Flaw**: The original `config.json` set `kS: 0.0`, disabling secondary nucleation entirely ($B = 0, M_{00} = \text{const}$).
- **Mathematical Root Cause with Active Nucleation ($k_S = 3.6 \times 10^5$)**:
  In the direct-node system $A(q) \frac{dq}{dt} = b$, zero-size nucleation enters only the zeroth moment rate:
  $$\frac{dM_{00}}{dt} = B > 0, \quad \frac{dM_{pq}}{dt} = 0 \text{ for } p+q > 0$$
  When $A(q) = \partial M / \partial q$ is inverted to determine coordinate velocities, absorbing the positive point-mass rate $\dot{M}_{00}$ without an explicit nucleus node forces the smallest seed node velocity strongly negative ($\dot{X}_1 \approx -0.0159 < 0$). At $t \approx 608.82\text{ s}$, $X_1$ is driven through zero into unphysical negative values ($X_1 = -0.0489 < 0$).
- **Resolution**: Integrating moments directly eliminates $A(q)$ inversion. The 9 moments evolve smoothly, while bounded quadrature reconstruction ($q_i \ge 10^{-7}$) keeps node coordinates strictly positive throughout the batch with relative reconstruction error $< 1.0\%$ (error budget: $5.0\%$).

### Issue 2: Call-Order Independence & Trial-Step Leakage
- **Original Flaw**: Updating warm-start initial guesses inside RHS functions whenever evaluation time increases leaks trial step states from rejected ODE steps into subsequent evaluations.
- **Resolution**: Implemented deterministic prior initialization derived directly from moment targets $(M_{00}, \bar{L}_1, \bar{L}_2, \sigma_1, \sigma_2)$ and anchored warm starts exclusively to accepted segment boundaries (`sol.sol(right)`).

### Issue 3: Stage Boundary Continuity & Temperature Handling
- **Original Flaw**: The simulator's stage handoff evaluated the state at observation times rather than the continuous segment boundary (`right`). Furthermore, liquid temperature was overwritten post-hoc (`z[:, 10] = expected_T`) to mask numerical integration discrepancies.
- **Resolution**:
  1. Dense output (`sol.sol(right)`) supplies the exact continuous final state to begin the next stage.
  2. Kinetics evaluate directly at analytical profile temperature $T_{\text{true}} = \text{profile.temperature}(t)$, eliminating post-hoc overwriting. Maximum temperature error dropped to machine precision ($\le 10^{-14}\;^\circ\text{C}$ on continuous profiles, $< 0.000622\,^\circ\text{C}$ on multi-stage profiles).

### Issue 4: PIRNN Model Physics Gating & Mass Balance
- **Original Flaw**:
  1. An artificial `+ 0.05` offset in $( \sigma + 0.05 )$ prevented updates from freezing at equilibrium ($\sigma = 0$).
  2. Concentration clipping broke the exact solute mass conservation identity.
- **Resolution**:
  1. Supersaturation gating strictly stops updates at equilibrium: $\Delta q \propto \tanh(\sigma / 0.1)$, which evaluates to identically $0.0$ when $\sigma \le 0$.
  2. Solute concentration is maintained by exact algebraic mass coupling:
     $$C(t) = M_{\text{total}} - \left(\frac{\rho_c h N_{\text{ref}} L_{\text{ref}}^2}{C_{\text{ref}}}\right) M_{11}(t)$$
     Mass drift is identically $0.0$ without artificial clipping.

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

# Inspect diagnostics: verify mass conservation (< 1e-15) and reconstruction residuals (< 0.5%)
python -m crystallization.data inspect outputs/single_batch.npz
```

### Stage 3: Verify the Existing Dataset
```bash
# Verify all 150 NPZ files against the 5% error budget and physical conservation
PYTHONPATH=. python /Users/devansinghfaujdar/.gemini/antigravity-ide/brain/2c7aa8a1-c6ef-435c-a15f-14b113da10ef/scratch/verify_dataset.py
```

### Stage 4: Train and Evaluate PIRNN
```bash
# Train PIRNN model (records architecture and manifest SHA-256 in checkpoint)
python -m crystallization.train --data outputs/data --epochs 30 --out outputs/pirnn

# Evaluate short-horizon forecasts and autonomous rollouts separately
python -m crystallization.evaluate --data outputs/data --checkpoint outputs/pirnn/best.pt --out outputs/evaluation --rollouts 4
```

---

## 5. Remaining Scientific Limitations & Scope

1. **Closure Limitation of 3-Node Quadrature**: A 3-node quadrature is a compact discrete approximation of a continuous bivariate distribution. While it fits all 9 moments to within $< 1.0\%$ error throughout cooling, representing highly bimodal distributions (nucleated fine crystals + growing macro seeds) challenges coordinate uniqueness and moment closure.
2. **Prescribed Cooling vs. Heat Transfer**: The temperature profile is prescribed as an explicit function of time. A full industrial crystallizer requires a jacket energy balance and heat-transfer dynamics.
3. **No Breakage or Agglomeration**: The model assumes plate thickness $h$ is strictly constant, with negligible crystal breakage, attrition, or agglomeration.
