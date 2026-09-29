# HAPPO

This is the standard two-agent HAPPO update: the UAV and UGV share one
centralized value critic and one joint GAE estimator, while actor policies are
updated sequentially in a randomized order.  A later actor's advantage is
multiplied by the exact new/old likelihood ratio of the actor already updated.

```bash
python -m du_iibtd_based_fading_delta.HAPPO.train --variant quant [common training options]
```
