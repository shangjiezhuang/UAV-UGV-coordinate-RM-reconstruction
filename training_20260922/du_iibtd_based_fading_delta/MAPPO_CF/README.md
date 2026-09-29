# MAPPO-CF

MAPPO-CF keeps MAPPO's decentralized UAV/UGV actors and centralized value
critic, and adds a centralized joint-action critic `Q(s, a_uav, a_ugv)`.
For each actor, the other actor's sampled action is fixed while all legal
actions of the current actor are marginalized to form a COMA-style
counterfactual baseline.  It does not use two independent GAEs.

```bash
python -m du_iibtd_based_fading_delta.MAPPO_CF.train --variant quant [common training options]
```
