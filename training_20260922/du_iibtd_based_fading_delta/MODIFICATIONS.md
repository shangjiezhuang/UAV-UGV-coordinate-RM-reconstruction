# Fading / uncertainty-delta variant

## 2026-09-20: production coupled 3% protocol and exact geometry acceleration

The current protocol is defined in `../README.md`: the sum of two signed
relative Frobenius-norm improvements below 3% switches local to global and
performs one full refit of received observations. Quant/noquant share both
the decision and reconstruction lifecycle. The former independent and
periodic refresh branches have been removed. The global hold remains 15 steps.

`run_paper_comparison_energy8500.sh` is the current launcher. Eight historical
launch scripts were removed from this copy; all remain in the frozen source
and verified backup. New saved configurations explicitly identify the coupled
protocol, and incompatible old configurations require explicit migration.

Reachability now uses scene-owned macro-action endpoint tables and bounded
caches. Crossing checks, wall stops, horizons and target-radius semantics are
preserved. Performance is measured against the same coupled 3% mechanism.
Per-update rollout and optimizer durations are recorded. See the verification
artifacts for equivalence tests and actual short-training timings.

The historical notes below describe earlier stages and do not override this
protocol.

This package is an isolated copy of `du_iibtd_based`. All algorithm folders are
included, while experiment outputs and Python caches are intentionally excluded.

The shared quantized and non-quantized environments contain the following changes:

1. Each sensing event uses the IIBTD observation disturbance
   `sum_r Phi[r, k] * eta[r, k] + epsilon[k]`. The default source-dependent
   small-scale fading standard deviation is `0.01`, and the fixed receiver-noise
   standard deviation is `0.01`. Both are configured in `UAVConfig` and use the
   episode RNG, matching the manuscript's `eta ~ N(0, sigma_eta^2)` and
   `epsilon ~ N(0, sigma_epsilon^2)` model.
2. Hybrid local-to-global switching uses the signed relative uncertainty
   improvement `(||U_prev||_F-||U_t||_F) / (||U_prev||_F + 1e-12)`. Positive
   values mean uncertainty decreased and negative values mean it increased.
   Two consecutive successful map updates below the default threshold `0.02`
   trigger global mode. The unsigned relative tensor delta remains available as
   a diagnostic, and the existing global hold and local re-entry rules are retained.
3. Quantization uses dataset-wide fixed raw endpoints `[0.0, 8.75]`. The upper
   endpoint exceeds the maximum of all eight training scenes plus six times the
   largest observation-noise standard deviation. For `log_first`, the endpoints
   are `log(1e-6)` and `log(8.75+1e-6)`. They do not depend on the current
   episode map.
4. The shared physical defaults are synchronized with the manuscript and frozen
   main-result configuration: 25 m building height, 9 kJ UAV energy budget, and
   8 Mbit unquantized reference payload per sensed band/window.

All remaining common training defaults follow the most recent complete run in
`du_iibtd_based_new/experiments/full_compare_logquant_h25_p2_8mbit_full_192k_120u_s42`:
the eight-scene suite (including RadioSeer index 1579), 192,000 transitions,
200-step rollouts, eight environments, six PPO epochs, progress/backtrack weight
2.0, remaining-time observations, quantization context for quant runs, 5% prefill
over a 200-window basis, and save/log/evaluation intervals of 10 updates.

`quant_range_analysis.json` records the dataset scan supporting the endpoints.
The temporary focused test module used during implementation was removed after
the full unit suite and all method-level smoke tests passed.

## 2026-09-14: remove effective communication bandwidth reduction

Communication uses the full allocated bandwidth:
`B_comm = current_comm_units * unit_bandwidth_hz`.
The former `0.8` factor is removed from actual channel evaluation and UGV
path-corridor service checks in both shared environments, and from the shared
two-path support controller's deterministic capacity estimate.

The latest complete result set,
`formal_runs/complete_qnq_cleanobs_h160_uimp008_192k_s42` (completed 2026-08-27),
was produced before this change and used the `0.8` factor. Its 26 result files
and existing checkpoints have not been regenerated. New results must use a
separate run directory to distinguish the updated communication model.

## 2026-09-14: remove the 0.2 sensing-bandwidth action

Both physical variants now default to bandwidth ratios `[0.3, 0.5, 0.6]`.
With 12 bandwidth units these allocate 4/6/8 units to sensing and 8/6/4
units to communication. The UAV action dimensions are 45 for quant and 15
for noquant. Legacy configuration files retain their original action spaces
when explicitly loaded; existing 60/20-action checkpoints cannot be reused
with the new three-choice configuration without adapting and retraining them.

`validate_clean160_contract.py` checks the three-choice protocol by default;
`--bandwidth-profile legacy-four` explicitly audits the earlier four-choice
protocol. The historical merge script and frozen results keep their original
60/20-action metadata. Use a new run directory for new training and evaluation.


## 2026-09-14: transmit power in the main experiment

Both physical variants now use RF output choices `[20, 27, 30]` dBm
(`[0.1, 0.501187, 1]` W), initially 27 dBm. UAV action order is
direction / bandwidth / quantization (quant only) / power, with power as the
fastest-changing index. The current action dimensions are **135 quant / 45
noquant**. All learned policies use the full categorical action space. Greedy
baselines use the lowest power predicted to clear the queue plus new sample,
or the highest power if none suffices, and repair choices against the mask.

Transmit energy is `P_tx_W * t_tx_seconds`. With a nonempty queue, duration is
`min(step_duration, queue_bits_before_tx / capacity_bps)` for a usable channel;
an outage charges a full-slot transmission attempt. Empty queues cost zero.
No amplifier, circuit, standby, or artificial energy multiplier is included.
The action mask reserves the selected power's full-slot energy for the current
step only; actual shorter transmissions refund this conservative allowance by
charging only their duration. Motion, sensing, and transmission are all included
in battery decrement and reported total energy. The mask may conservatively
exclude a partially affordable action before channel realization.

Channel calculations and service-target capacity caches use the selected power.
Evaluation traces include power index, dBm, watts, duration, and TX joules;
episode records and aggregate metrics include TX energy. No additional reward
coefficient or observation dimension was introduced.

CLI: `--tx_power_choices_dbm 20 27 30 --tx_power_dbm 27`; a singleton choice
supports fixed-power ablations. Current clean160 validation checks all three
power choices, enabled TX energy, and 135/45 actions. To audit old results use
`--power-profile legacy-fixed`, together with `--bandwidth-profile legacy-four`
for the old four-ratio protocol. Loading historical configs warns explicitly,
preserves their singleton fixed power and original zero-TX-energy convention;
new training uses the new defaults. Retrain old checkpoints for the new action
space. Earlier source archives are required for exact historical reproduction.

`run_main_power_clean160.sh` prepares all 20 learned methods and six heuristic
baselines under a new `main_power202730_bw3_cleanobs_h160_uimp008_192k_s42`
directory. It does not reuse historical configurations/checkpoints/results or
call the historical hard-coded merge script. Full training has not been started.

Backup immediately before this change (129 source files, hashes verified):
`/home/zsj/works/work1/code/.change_backups/before_main_power_20260914_232639/source_code.tar.gz`.
Validation: 67 focused regression tests, real-scene evaluation with all power
levels and both variants, all ten UGV control modes in both variants, and one
four-transition training update with checkpoint save for each of the 20 learned
variant/trainer combinations. These short checks establish functionality, not
convergence or comparative performance.

The 9000 J budget, noise figure 8 dB, UAV height 30 m and building height 25 m
remain unchanged. Fixed-trajectory sensitivity calculations are in
`analysis/uav_power_review_20260914/`. On the current 200 m scene, even the
lowest new power cannot suffer the deterministic -5 dB outage threshold under
the existing loss model (whole-map worst case ~0.44 dB). Higher buildings alone
do not resolve this. Extra equivalent noise of 15 or 20 dB was explored as a
diagnostic scenario only; this is not a measured receiver-noise specification.
Do not interpret these replays as newly trained results.


## 2026-09-14: reproducible building heights uniformly between 25 and 29 m

The main experiment now uses `scene.building_height_mode=uniform_per_building`,
`building_height_min_m=25`, `building_height_max_m=29`, and
`building_height_seed=42`. Each four-connected component of the binary building
mask receives one continuous uniform height. Dataset masks have no building
instance IDs, so side-touching buildings are treated as one connected footprint;
diagonally touching footprints remain separate. Non-building height is zero.

Heights use an independent PCG64 stream seeded by the geometry seed and a stable
SHA-256 digest of the footprint mask. They are identical for quant/noquant,
training seeds, episode resets, worker order, and relocated dataset paths.
Changing the geometry seed creates another reproducible building realization.
Simulation data generation and the direct GridScene fallback share one helper.
Explicit supplied height maps remain authoritative. Geometry affects the
communication LoS check; the stored RadioSeer sensing truth is not regenerated.

Config export records mode, range and seed. Training CLI supports the same four
fields. `--building_height_mode fixed --building_height_m 25` supports the earlier
constant-height geometry. Loading older saved configs automatically preserves
their fixed heights with a warning. The formal protocol validator now requires
random heights unless `--height-profile legacy-fixed` is explicitly selected.

The main launcher writes to the new directory
`formal_runs/main_power202730_bw3_h25to29_cleanobs_h160_uimp008_192k_s42`.
Power actions remain 20/27/30 dBm, the UAV stays at 30 m and energy budget at
9000 J. Full training was not started. The existing nominal path-loss outage
gate and 8 dB receiver noise figure are unchanged; an additional 15 dB noise
level was analyzed only as an external-interference scenario, not adopted.

Backup before this change (132 source files, verified):
`/home/zsj/works/work1/code/.change_backups/before_random_heights_20260914_235150/source_code.tar.gz`.
Validation: 74 regression and geometry tests passed; all eight real scene height
maps agree across variants and different training seeds; per-building constancy,
range, reset stability, LoS effects, and real environment energy accounting pass.
Sensitivity outputs and assumptions are under `analysis/random_heights_snr_20260914`.
They compare nominal and shadow-inclusive SNR on fixed old trajectories and do
not establish the performance of newly trained policies.

## 2026-09-15: integer building heights and joint-height diagnostics

The current default is now `uniform_integer_per_building`: each four-connected
building footprint draws one value uniformly from {25, 26, 27, 28, 29} metres.
The independent geometry seed remains 42. The previous continuous mode remains
available for saved configurations; fixed historical configurations retain their
original heights. Integer mode validates integer-valued bounds, including the
inclusive upper bound. The protocol validator accepts `legacy-continuous` only
when that historical profile is explicitly requested.

LoS already reads the per-cell height map and compares the interpolated UAV-UGV
ray height with the building height along the rasterized path. Results for the
same endpoint pair are cached. Integer heights require no different LoS rule.

The main launcher now uses
`formal_runs/main_power202730_bw3_h25to29int_cleanobs_h160_uimp008_192k_s42`.
Validation: all 76 regression/geometry tests passed, all eight scene height maps
agree across variants and training seeds, real environment resets and power
steps pass, and all 20 learned-method configurations satisfy the main protocol.
No full training was started. UAV altitude 30 m, receiver noise figure 8 dB,
nominal SNR outage threshold -5 dB, power actions 20/27/30 dBm, and the 9000 J
budget remain unchanged.

Backup immediately before this refinement (134 source files, hashes verified):
`/home/zsj/works/work1/code/.change_backups/before_integer_heights_20260915_000344/source_code.tar.gz`.
The change diff and applied hashes are stored alongside the backup. Diagnostic
data are under `analysis/integer_heights_snr_20260915/`. They replay frozen old
trajectories; they are not newly trained performance or an independent final
test set. Raising both UAV/building heights by the same factor preserves LoS
when UGV height is zero; adding the same offset can increase obstruction. In
this model, NLoS adds a fixed 15 dB and taller buildings do not add further loss
once blocked. The tested joint-height scenarios still have zero nominal-gated
outage at 20 dBm. Additional 15 dB equivalent noise is a diagnostic interference
scenario only and has not been adopted as a receiver specification.

## 2026-09-15: received-SNR transmission outage with nominal planning

Actual transmission now defaults to `comm.outage_snr_mode=received`: the same
shadow-inclusive `snr_db` used by Shannon capacity is compared with the existing
-5 dB service threshold. Capacity is zero below the threshold; equality permits
service. One shadow draw per channel call is retained. The historical field
`large_scale_snr_db` still reports FSPL plus fixed NLoS excess only; shadowing
itself is also a large-scale phenomenon, so this field is a nominal-link proxy.

Candidate planning remains deterministic and uses nominal path loss/SNR. The
controller and environment planner code were not changed. Recovery continues
to react to observed outage/queue state, now including outages caused by the
realized shadow loss. An attempted transmission during outage with a nonempty
queue still charges P_tx times the full slot, while transmitting zero bits.

Both physical variants expose `--outage_snr_mode received|nominal`. Loading a
saved communication config without this field preserves its historical nominal
gate and emits a warning. The formal validator requires received mode by default;
`--outage-profile legacy-nominal` is required to audit historical results. The
main launcher explicitly selects received mode and writes to
`formal_runs/main_power202730_bw3_h25to29int_rxsnr_cleanobs_h160_uimp008_192k_s42`.

Backup before this change (134 source files, hashes verified):
`/home/zsj/works/work1/code/.change_backups/before_received_snr_20260915_001952/source_code.tar.gz`.
Validation: all 83 tests passed, including received/nominal threshold crossings,
threshold equality, rate consistency, unchanged RNG sequence, deterministic
planning, CLI validation and saved-config compatibility. Forced-channel checks
in real quant/noquant environments verified stalled transmission, unchanged
queued bits during outage, charged TX energy, recovery entry, and resumed
service. All 20 main learned-method configs passed the formal protocol; nominal
mode was rejected under the current profile and historical profiles passed.
The launcher passes bash syntax validation after normalizing Linux newlines.

UAV/building heights, 20/27/30 dBm power choices, 9000 J budget, 8 dB noise figure,
and the existing 15 dB NLoS excess remain unchanged. Full training was not
started. `analysis/received_snr_20260915/` contains functional-check results and
fixed-trajectory sensitivity to literature example excess-loss pairs (LoS/NLoS:
1/20 and 1.6/23 dB). These pairs were not adopted in main code or calibrated to
the current 3.5 GHz maps. Fixed historical trajectories are diagnostic only.
## 2026-09-15: 50 m UAV, integer 45-48 m buildings, 9/12/15 dBm

The user requested a base building range of integer 25-28 m, then a 20 m
increase to both the UAV and buildings. The resulting geometry is UAV 50 m
and per-building heights sampled uniformly from {45, 46, 47, 48} m. UGV
height stays 0 m. The geometry seed remains 42 and the existing connected
footprint assignment is shared across quantized/nonquantized variants.

Both shared configurations and the main launcher now use RF power actions
{9, 12, 15} dBm, initially 12 dBm. RF energy remains P_tx * t_tx and the
9000 J energy budget remains in place. Bandwidth actions, receiver noise,
shadowing parameters, and received-SNR outage versus nominal planning are
preserved. Action counts remain 135 (quant) and 45 (noquant).

The current protocol validator checks UAV/UGV altitude, building range,
and the new power set. Explicit `--height-profile legacy-integer` and
`--power-profile legacy-202730` retain auditing of the immediately previous
settings; earlier fixed/continuous geometry and fixed-power modes remain.
Existing saved configurations keep their stored physical parameters.

The main experiment now writes to
`formal_runs/main_power091215_bw3_uav50_h45to48int_rxsnr_cleanobs_h160_uimp008_192k_s42`.
This parameter update does not launch full training.

Pre-change source backup (135 files, contents checked against SHA-256):
`/home/zsj/works/work1/code/.change_backups/before_height50_power091215_20260915_011036/source_code.tar.gz`.


## 2026-09-15: paper comparison at 8500 J

The user chose to retain the abstract 12 W flight / 8 W hover model for
algorithm comparison after reviewing literature reference platforms. Both
variants now default to 8500 J; sensing remains band-scaled up to 5 W and
RF transmission remains P_tx * actual attempt duration at 9/12/15 dBm.
These parameters are simulation assumptions, not a calibrated aircraft model.
The 160-decision energy upper bound is 8485.059644 J, so 8500 J is not a
tight energy constraint. No claim of budget-induced power optimization is made.

The requested run is run_paper_comparison_energy8500.sh, scoped to the
methods listed in paper/realMain.tex: Proposed, Fixed-UGV, Target-Only A*,
MAPPO, IPPO, HAPPO, Greedy-A*, and Random, each quantized/non-quantized.
This means 12 learned runs and four non-learning baseline evaluations, using
eight scenes, 192000 training transitions per learned run and 160 decisions
per episode. The full 26-method launcher is available but is not started.
Outputs are isolated under formal_runs/paper_power091215_bw3_uav50_h45to48int_rxsnr_e8500_cleanobs_h160_uimp008_192k_s42.
The shared contract checks the energy coefficients and budget; previous 9000 J
records require --energy-profile legacy-9000. Training/evaluation seeds,
geometry, power choices, and received-SNR outage remain as configured before.
The known no-affordable-action behavior at tighter budgets is not changed in
this parameter-only run; the verified 8500 J bound prevents it before horizon.

Backup: /home/zsj/works/work1/code/.change_backups/before_energy8500_20260915_014142/source_code.tar.gz

## 2026-09-15: isolated urban LoS/NLoS excess loss 1/20 dB

The user requested urban mean excess losses of 1 dB (LoS) and 20 dB (NLoS),
and explicitly asked to keep the ongoing training running. This version lives
under `code/experiment_versions/urban_los1_nlos20_20260915`; the live package
remains at its original path with the original 0/15 dB configuration.

Both physical variants, their corridor-recovery link estimates, and the shared
two-path A* controller now use the same 1/20 dB pair above FSPL. The two losses
are alternatives selected by geometric LoS, not cumulative losses. CLI flags,
saved configuration, console summaries, and the formal protocol validator all
record/check both values. Actual transmission still uses received SNR including
shadowing; future candidate planning still uses nominal SNR.

Receiver noise stays -174 dBm/Hz + 10*log10(B_comm/Hz) + 8 dB NF. Source load
stays 8 Mbit/band at 32 reference bits; shared bandwidth stays 50 MHz; queue
stays 512 Mbit; energy stays 8500 J; power choices stay 9/12/15 dBm; UAV/building
heights stay 50 m / blockwise integer 45-52 m with split threshold 1200 m^2.

Old saved configs missing los_excess_db explicitly restore zero LoS excess
loss and their saved NLoS value with a compatibility warning. Formal checks
reject 0/15 dB unless --loss-profile legacy-0-15 is specified. The new paper
launcher resolves its own source root and names results paper_urban_los1_nlos20_*.

Validation: 81 unit tests passed, including hand-calculated LoS/NLoS budgets,
all three communication-bandwidth noise powers, both planners, outage/rate
consistency, and historical config reproduction. All 12 learned method configs
resolve to 1/20 dB. Both quant/noquant CPU smoke runs completed two updates
(16 transitions each), using real scene 8513 and the configured reconstruction
backend, and saved final checkpoints with the new parameters. Smoke runs are
functional checks, not performance estimates. No new formal training started.

Pre-change backup (138 source files; SHA-256 verified):
`/home/zsj/works/work1/code/.change_backups/before_urban_los1_nlos20_20260915_224811/source_code.tar.gz`.
After validation all live source hashes still match this backup and driver
21299 remains alive. Verification records are in the version root/verification.

## Obstruction-length correction (2026-09-15)

- NLOS now uses 20 + min(0.1 * horizontal blocked length in metres, 12) dB; LOS stays 1 dB.
- Shadow standard deviations remain 2/6 dB. Geometry correction is deterministic; received-SNR outage still uses the shadow sample.
- One shared loss helper is used by both physical variants, actual service, support capacity scoring and recovery corridor planning.
- Length uses the existing center-line roof test and a bounded per-scene geometry cache, independent of artificial building IDs.
- Saved configs without length parameters explicitly disable the new correction. Formal validation distinguishes fixed 1/20, fixed 0/15 and the new length protocol.
- Final evaluation traces include blocked length, length correction and total excess loss for both variants.
- Formal launcher output prefix includes len01cap12. The active old training source remains unchanged.
- Parameters are exploratory and not a calibrated material-penetration model.
