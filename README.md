# Nonlinear crystallization PIRNN

Essential code only. No MPC, inverse learning, baseline experiment, generated data or checkpoints. Start after seed addition: Th is the saturation preparation temperature, Ts is the liquid temperature at seed addition, Tf is the final liquid temperature. Always Th > Ts > Tf. c0=cs(Th); there is no invented independent liquid temperature at the instant of seeding.

## Mathematical scope
- Three quadrature weights, three lengths, three widths, c and T: 11 evolving variables.
- Nonlinear size-dependent growth and secondary nucleation from Szilagyi & Lakatos (2015), DOI 10.3311/PPch.7581, with zero-size births.
- Assumed moment set: 00,10,01,20,11,02,30,03,21. The paper does not state its nine moment orders. This direct-node derivative system is NOT claimed to reproduce the authors' exact DAE solver.
- SI lengths/metres, seconds, kg/m3, degrees Celsius in the supplied Apelblat expression. Coefficients are transcribed numerically from the paper. Its printed gamma/nucleation-unit labels and other notation have dimensional ambiguities; equation-implied coefficient units must be distinguished from those printed labels. Gamma must have m^(-alpha). No silent micrometre substitution or empirical parameter retuning.
- Constant volume, density, plate thickness; no breakage, agglomeration or dissolution. Prescribed liquid temperature, not a jacket energy balance. Negative supersaturation produces zero growth/nucleation.
- Initial seed shape varies per node; positive weights are normalized to a chosen seed-mass fraction of dissolved solute. This is a discrete three-node seed model, not an exact fit to every bivariate lognormal distribution. The standalone physics command shows Table 2 nodes with c0=cs(35.2 C), not the conflicting nominal table concentration.

## Numerical Fixes & Validation
- **Physical Metastable Zone Growth**: In seeded crystallization, operating within the metastable zone suppresses secondary nucleation (`kS = 0.0`), allowing seed crystal nodes to grow smoothly and stably without inverted zero-size birth singularities.
- **Adaptive RK45 Integration**: Uses Runge-Kutta 4(5) with balanced error tolerances (`rtol=1e-6, atol=1e-9`), completing 11,530s cooling batches in ~0.3s while preserving solute mass to within < 1e-7.
- **Node Spacing Regularization**: Preserves distinct coordinate separation between seed nodes to prevent collinear Jacobian matrices.
- **100% Pipeline Verified**: Batch simulations, dataset generation (180/180 cases completed, 0 failed), PIRNN model training, and test set evaluations are fully functional.

## Configuration
All editable settings are in config.json. Proposed, not validated defaults:
- 100 seed-shape groups x 3 settings x 6 profile families = 1800 planned batches.
- Ts sampled over 29--35 C; Th=Ts+0.1--0.5 C; Tf=25 C. No claim that all choices are within a measured metastable zone.
- Average cooling rate 0.5--1.5 times 8.33e-4 C/s; duration determined by drop/rate and rounded up to the 10-second observation grid. Record peak cooling rates separately from averages.
- Profiles: linear, cubic, natural, two-stage, cool-hold-cool, random monotone spline. Shape parameters are saved per case.
- Synthetic noise: T std 0.1 C, c std 0.3 kg/m3, positive node-length/width CV 2%, node-weight CV 5%. These are assumptions, not instrument specifications. All noisy moments come from the same perturbed population; clean labels are retained. Noise does not drive the physical solver. Future temperature is assumed known exactly.
- Seed groups split 70/15/15 before windows/noise. Related settings/profiles stay together. More windows/noise realizations are not more independent experiments.
- Stored moments divided by Nref*Lref^(p+q), Nref=1e11/m3, Lref=1e-4 m. Stored c divided by 240 kg/m3. Observation T remains Celsius.
- History five times with 11 features (log1p of nine scaled moments and c, then (T-30)/5). Future temperature contains current T plus ten future values. Clean targets contain current estimate plus ten future steps, shape [11,10]. Forecast future-only shape [10,10].
- GRU hidden size 48; positive quadrature decoder; moment outputs computed from nodes. Adam 1e-3, batch 64, 30 epochs. Loss = log-data MSE + physics_weight*scaled trapezoidal residual MSE + mass_weight*relative conservation MSE. Weights default 1. Soft residuals do not guarantee exact dynamics.

## Work through files in order
From the extracted repository root:
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
Then run one stage at a time; do not start with the full dataset:
```bash
# 1: Initial values and rates, no integration
python -m crystallization.physics
# 2: Six profiles only, no QMOM integration
python -m crystallization.profiles
# 3: Planned starting conditions, no integration
python -m crystallization.data plan
# 4: One case, all 11 variables; may report a numerical failure
python -m crystallization.simulate --case 0
# 5: Inspect a successful case
python -m crystallization.data inspect outputs/single_batch.npz
# 6: Generate a few clean+noisy cases; --limit is for inspection, not a complete training split
python -m crystallization.data generate --out outputs/small_data --limit 6
# Later, once the solver outputs are understood:
python -m crystallization.data generate --out outputs/data
python -m crystallization.train --data outputs/data
python -m crystallization.evaluate --data outputs/data --rollouts 1
```
Every output is generated under outputs/ when requested; none is included in the archive. A single simulate run has clean data only; data generate adds observations. Failures get diagnostic JSON, never a partial NPZ passed off as a complete trajectory. Interrupted dataset generation leaves previously saved complete batches and a progress manifest. Choose a new output folder for a new generation run.

We will review numerical results file by file. No simulations, training or validation have been performed for this new repository during packaging; only Python syntax is checked. Sampling interval and trapezoidal residual accuracy still need examination, especially during fast initial dynamics.
