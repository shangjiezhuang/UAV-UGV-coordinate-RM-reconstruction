# Fading / uncertainty-delta variant

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
