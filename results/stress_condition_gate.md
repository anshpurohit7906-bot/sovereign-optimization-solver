# Stress Test: Condition-Gate Replacement

**Goal:** Validate sparse cond1 estimator preserves gate safety.

## Gate Logic
```
if min_xB >= -tol:    -> no_action
elif min_xB >= -eff_tol: -> soft_clamp
else:                    -> repair
eff_tol = max(kappa*eps*||b||_inf*50.0, 1e-07)
```

## Part 1: Real-Basis Gate Parameters

| model | iter | min_xB | cond2 | cond1 | raw_exact | raw_est | eff_exact | eff_est |
|-------|------|--------|-------|-------|-----------|---------|-----------|---------|
| pilot4_plain | 0 | -1.044e-15 | 1.251e+05 | 1.268e+06 | 3.690e-05 | 3.741e-04 | 3.690e-05 | 3.741e-04 |
| pilot4_plain | 1500 | -3.940e-14 | 1.354e+05 | 2.306e+06 | 3.994e-05 | 6.804e-04 | 3.994e-05 | 6.804e-04 |
| pilot4_plain | 2250 | -6.328e-15 | 1.355e+05 | 2.231e+06 | 3.999e-05 | 6.584e-04 | 3.999e-05 | 6.584e-04 |
| pilot4_plain | 3000 | -8.993e-15 | 1.257e+05 | 1.667e+06 | 3.709e-05 | 4.917e-04 | 3.709e-05 | 4.917e-04 |
| pilot4_plain | 750 | -1.117e-15 | 1.353e+05 | 1.457e+06 | 3.992e-05 | 4.300e-04 | 3.992e-05 | 4.300e-04 |
| pilot87 | 0 | 0.000e+00 | 2.831e+04 | 1.053e+07 | 5.178e-06 | 1.927e-03 | 5.178e-06 | 1.927e-03 |
| pilot87 | 12500 | -0.000e+00 | 1.133e+05 | 3.244e+07 | 2.072e-05 | 5.934e-03 | 2.072e-05 | 5.934e-03 |
| pilot87 | 18750 | 0.000e+00 | 2.376e+05 | 4.678e+07 | 4.346e-05 | 8.556e-03 | 4.346e-05 | 8.556e-03 |
| pilot87 | 25000 | -0.000e+00 | 7.165e+04 | 2.753e+07 | 1.310e-05 | 5.035e-03 | 1.310e-05 | 5.035e-03 |
| pilot87 | 6250 | 0.000e+00 | 2.060e+04 | 7.432e+06 | 3.768e-06 | 1.359e-03 | 3.768e-06 | 1.359e-03 |

## Part 2: Gate-Threshold Sweep

| model | iter | test_point | min_xB | exact | est | match | conservatism |
|-------|------|------------|--------|-------|-----|-------|--------------|
| pilot4_plain | 0 | 0 | 0.000e+00 | no_action | no_action | Y | SAME |
| pilot4_plain | 0 | -tol | -1.000e-07 | no_action | no_action | Y | SAME |
| pilot4_plain | 0 | -1.1*tol | -1.100e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | -0.1*ee | -3.690e-06 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | -ee | -3.690e-05 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | -1.1*ee | -4.059e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 0 | -10*ee | -3.690e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 0 | -0.1*es | -3.741e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 0 | -es | -3.741e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 0 | -1.1*es | -4.115e-04 | repair | repair | Y | SAME |
| pilot4_plain | 0 | -10*es | -3.741e-03 | repair | repair | Y | SAME |
| pilot4_plain | 1500 | 0 | 0.000e+00 | no_action | no_action | Y | SAME |
| pilot4_plain | 1500 | -tol | -1.000e-07 | no_action | no_action | Y | SAME |
| pilot4_plain | 1500 | -1.1*tol | -1.100e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | -0.1*ee | -3.994e-06 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | -ee | -3.994e-05 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | -1.1*ee | -4.393e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 1500 | -10*ee | -3.994e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 1500 | -0.1*es | -6.804e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 1500 | -es | -6.804e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 1500 | -1.1*es | -7.484e-04 | repair | repair | Y | SAME |
| pilot4_plain | 1500 | -10*es | -6.804e-03 | repair | repair | Y | SAME |
| pilot4_plain | 2250 | 0 | 0.000e+00 | no_action | no_action | Y | SAME |
| pilot4_plain | 2250 | -tol | -1.000e-07 | no_action | no_action | Y | SAME |
| pilot4_plain | 2250 | -1.1*tol | -1.100e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | -0.1*ee | -3.999e-06 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | -ee | -3.999e-05 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | -1.1*ee | -4.399e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 2250 | -10*ee | -3.999e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 2250 | -0.1*es | -6.584e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 2250 | -es | -6.584e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 2250 | -1.1*es | -7.243e-04 | repair | repair | Y | SAME |
| pilot4_plain | 2250 | -10*es | -6.584e-03 | repair | repair | Y | SAME |
| pilot4_plain | 3000 | 0 | 0.000e+00 | no_action | no_action | Y | SAME |
| pilot4_plain | 3000 | -tol | -1.000e-07 | no_action | no_action | Y | SAME |
| pilot4_plain | 3000 | -1.1*tol | -1.100e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | -0.1*ee | -3.709e-06 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | -ee | -3.709e-05 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | -1.1*ee | -4.079e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 3000 | -10*ee | -3.709e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 3000 | -0.1*es | -4.917e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 3000 | -es | -4.917e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 3000 | -1.1*es | -5.409e-04 | repair | repair | Y | SAME |
| pilot4_plain | 3000 | -10*es | -4.917e-03 | repair | repair | Y | SAME |
| pilot4_plain | 750 | 0 | 0.000e+00 | no_action | no_action | Y | SAME |
| pilot4_plain | 750 | -tol | -1.000e-07 | no_action | no_action | Y | SAME |
| pilot4_plain | 750 | -1.1*tol | -1.100e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | -0.1*ee | -3.992e-06 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | -ee | -3.992e-05 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | -1.1*ee | -4.392e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 750 | -10*ee | -3.992e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 750 | -0.1*es | -4.300e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 750 | -es | -4.300e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot4_plain | 750 | -1.1*es | -4.730e-04 | repair | repair | Y | SAME |
| pilot4_plain | 750 | -10*es | -4.300e-03 | repair | repair | Y | SAME |
| pilot87 | 0 | 0 | 0.000e+00 | no_action | no_action | Y | SAME |
| pilot87 | 0 | -tol | -1.000e-07 | no_action | no_action | Y | SAME |
| pilot87 | 0 | -1.1*tol | -1.100e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | -0.1*ee | -5.178e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | -ee | -5.178e-06 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | -1.1*ee | -5.696e-06 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 0 | -10*ee | -5.178e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 0 | -0.1*es | -1.927e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 0 | -es | -1.927e-03 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 0 | -1.1*es | -2.119e-03 | repair | repair | Y | SAME |
| pilot87 | 0 | -10*es | -1.927e-02 | repair | repair | Y | SAME |
| pilot87 | 12500 | 0 | 0.000e+00 | no_action | no_action | Y | SAME |
| pilot87 | 12500 | -tol | -1.000e-07 | no_action | no_action | Y | SAME |
| pilot87 | 12500 | -1.1*tol | -1.100e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | -0.1*ee | -2.072e-06 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | -ee | -2.072e-05 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | -1.1*ee | -2.279e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 12500 | -10*ee | -2.072e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 12500 | -0.1*es | -5.934e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 12500 | -es | -5.934e-03 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 12500 | -1.1*es | -6.527e-03 | repair | repair | Y | SAME |
| pilot87 | 12500 | -10*es | -5.934e-02 | repair | repair | Y | SAME |
| pilot87 | 18750 | 0 | 0.000e+00 | no_action | no_action | Y | SAME |
| pilot87 | 18750 | -tol | -1.000e-07 | no_action | no_action | Y | SAME |
| pilot87 | 18750 | -1.1*tol | -1.100e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | -0.1*ee | -4.346e-06 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | -ee | -4.346e-05 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | -1.1*ee | -4.780e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 18750 | -10*ee | -4.346e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 18750 | -0.1*es | -8.556e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 18750 | -es | -8.556e-03 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 18750 | -1.1*es | -9.411e-03 | repair | repair | Y | SAME |
| pilot87 | 18750 | -10*es | -8.556e-02 | repair | repair | Y | SAME |
| pilot87 | 25000 | 0 | 0.000e+00 | no_action | no_action | Y | SAME |
| pilot87 | 25000 | -tol | -1.000e-07 | no_action | no_action | Y | SAME |
| pilot87 | 25000 | -1.1*tol | -1.100e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | -0.1*ee | -1.310e-06 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | -ee | -1.310e-05 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | -1.1*ee | -1.442e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 25000 | -10*ee | -1.310e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 25000 | -0.1*es | -5.035e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 25000 | -es | -5.035e-03 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 25000 | -1.1*es | -5.538e-03 | repair | repair | Y | SAME |
| pilot87 | 25000 | -10*es | -5.035e-02 | repair | repair | Y | SAME |
| pilot87 | 6250 | 0 | 0.000e+00 | no_action | no_action | Y | SAME |
| pilot87 | 6250 | -tol | -1.000e-07 | no_action | no_action | Y | SAME |
| pilot87 | 6250 | -1.1*tol | -1.100e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | -0.1*ee | -3.768e-07 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | -ee | -3.768e-06 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | -1.1*ee | -4.145e-06 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 6250 | -10*ee | -3.768e-05 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 6250 | -0.1*es | -1.359e-04 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 6250 | -es | -1.359e-03 | repair | soft_clamp | N | LESS_conservative |
| pilot87 | 6250 | -1.1*es | -1.495e-03 | repair | repair | Y | SAME |
| pilot87 | 6250 | -10*es | -1.359e-02 | repair | repair | Y | SAME |

## Part 3: Adversarial b_norm Scaling

| model | iter | scale | eff_exact | eff_est | ratio | exact | est | match | conservatism |
|-------|------|-------|-----------|---------|-------|-------|-----|-------|--------------|
| pilot4_plain | 0 | 1e+00 | 3.690e-05 | 3.741e-04 | 0.0986 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | 1e+00 | 3.690e-05 | 3.741e-04 | 0.0986 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | 1e+03 | 3.690e-02 | 3.741e-01 | 0.0986 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | 1e+03 | 3.690e-02 | 3.741e-01 | 0.0986 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | 1e+06 | 3.690e+01 | 3.741e+02 | 0.0986 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | 1e+06 | 3.690e+01 | 3.741e+02 | 0.0986 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | 1e+09 | 3.690e+04 | 3.741e+05 | 0.0986 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | 1e+09 | 3.690e+04 | 3.741e+05 | 0.0986 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | 1e+12 | 3.690e+07 | 3.741e+08 | 0.0986 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 0 | 1e+12 | 3.690e+07 | 3.741e+08 | 0.0986 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | 1e+00 | 3.994e-05 | 6.804e-04 | 0.0587 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | 1e+00 | 3.994e-05 | 6.804e-04 | 0.0587 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | 1e+03 | 3.994e-02 | 6.804e-01 | 0.0587 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | 1e+03 | 3.994e-02 | 6.804e-01 | 0.0587 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | 1e+06 | 3.994e+01 | 6.804e+02 | 0.0587 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | 1e+06 | 3.994e+01 | 6.804e+02 | 0.0587 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | 1e+09 | 3.994e+04 | 6.804e+05 | 0.0587 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | 1e+09 | 3.994e+04 | 6.804e+05 | 0.0587 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | 1e+12 | 3.994e+07 | 6.804e+08 | 0.0587 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 1500 | 1e+12 | 3.994e+07 | 6.804e+08 | 0.0587 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | 1e+00 | 3.999e-05 | 6.584e-04 | 0.0607 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | 1e+00 | 3.999e-05 | 6.584e-04 | 0.0607 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | 1e+03 | 3.999e-02 | 6.584e-01 | 0.0607 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | 1e+03 | 3.999e-02 | 6.584e-01 | 0.0607 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | 1e+06 | 3.999e+01 | 6.584e+02 | 0.0607 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | 1e+06 | 3.999e+01 | 6.584e+02 | 0.0607 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | 1e+09 | 3.999e+04 | 6.584e+05 | 0.0607 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | 1e+09 | 3.999e+04 | 6.584e+05 | 0.0607 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | 1e+12 | 3.999e+07 | 6.584e+08 | 0.0607 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 2250 | 1e+12 | 3.999e+07 | 6.584e+08 | 0.0607 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | 1e+00 | 3.709e-05 | 4.917e-04 | 0.0754 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | 1e+00 | 3.709e-05 | 4.917e-04 | 0.0754 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | 1e+03 | 3.709e-02 | 4.917e-01 | 0.0754 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | 1e+03 | 3.709e-02 | 4.917e-01 | 0.0754 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | 1e+06 | 3.709e+01 | 4.917e+02 | 0.0754 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | 1e+06 | 3.709e+01 | 4.917e+02 | 0.0754 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | 1e+09 | 3.709e+04 | 4.917e+05 | 0.0754 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | 1e+09 | 3.709e+04 | 4.917e+05 | 0.0754 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | 1e+12 | 3.709e+07 | 4.917e+08 | 0.0754 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 3000 | 1e+12 | 3.709e+07 | 4.917e+08 | 0.0754 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | 1e+00 | 3.992e-05 | 4.300e-04 | 0.0929 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | 1e+00 | 3.992e-05 | 4.300e-04 | 0.0929 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | 1e+03 | 3.992e-02 | 4.300e-01 | 0.0929 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | 1e+03 | 3.992e-02 | 4.300e-01 | 0.0929 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | 1e+06 | 3.992e+01 | 4.300e+02 | 0.0929 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | 1e+06 | 3.992e+01 | 4.300e+02 | 0.0929 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | 1e+09 | 3.992e+04 | 4.300e+05 | 0.0929 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | 1e+09 | 3.992e+04 | 4.300e+05 | 0.0929 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | 1e+12 | 3.992e+07 | 4.300e+08 | 0.0929 | soft_clamp | soft_clamp | Y | SAME |
| pilot4_plain | 750 | 1e+12 | 3.992e+07 | 4.300e+08 | 0.0929 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | 1e+00 | 5.178e-06 | 1.927e-03 | 0.0027 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | 1e+00 | 5.178e-06 | 1.927e-03 | 0.0027 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | 1e+03 | 5.178e-03 | 1.927e+00 | 0.0027 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | 1e+03 | 5.178e-03 | 1.927e+00 | 0.0027 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | 1e+06 | 5.178e+00 | 1.927e+03 | 0.0027 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | 1e+06 | 5.178e+00 | 1.927e+03 | 0.0027 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | 1e+09 | 5.178e+03 | 1.927e+06 | 0.0027 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | 1e+09 | 5.178e+03 | 1.927e+06 | 0.0027 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | 1e+12 | 5.178e+06 | 1.927e+09 | 0.0027 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 0 | 1e+12 | 5.178e+06 | 1.927e+09 | 0.0027 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | 1e+00 | 2.072e-05 | 5.934e-03 | 0.0035 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | 1e+00 | 2.072e-05 | 5.934e-03 | 0.0035 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | 1e+03 | 2.072e-02 | 5.934e+00 | 0.0035 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | 1e+03 | 2.072e-02 | 5.934e+00 | 0.0035 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | 1e+06 | 2.072e+01 | 5.934e+03 | 0.0035 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | 1e+06 | 2.072e+01 | 5.934e+03 | 0.0035 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | 1e+09 | 2.072e+04 | 5.934e+06 | 0.0035 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | 1e+09 | 2.072e+04 | 5.934e+06 | 0.0035 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | 1e+12 | 2.072e+07 | 5.934e+09 | 0.0035 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 12500 | 1e+12 | 2.072e+07 | 5.934e+09 | 0.0035 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | 1e+00 | 4.346e-05 | 8.556e-03 | 0.0051 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | 1e+00 | 4.346e-05 | 8.556e-03 | 0.0051 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | 1e+03 | 4.346e-02 | 8.556e+00 | 0.0051 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | 1e+03 | 4.346e-02 | 8.556e+00 | 0.0051 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | 1e+06 | 4.346e+01 | 8.556e+03 | 0.0051 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | 1e+06 | 4.346e+01 | 8.556e+03 | 0.0051 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | 1e+09 | 4.346e+04 | 8.556e+06 | 0.0051 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | 1e+09 | 4.346e+04 | 8.556e+06 | 0.0051 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | 1e+12 | 4.346e+07 | 8.556e+09 | 0.0051 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 18750 | 1e+12 | 4.346e+07 | 8.556e+09 | 0.0051 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | 1e+00 | 1.310e-05 | 5.035e-03 | 0.0026 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | 1e+00 | 1.310e-05 | 5.035e-03 | 0.0026 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | 1e+03 | 1.310e-02 | 5.035e+00 | 0.0026 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | 1e+03 | 1.310e-02 | 5.035e+00 | 0.0026 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | 1e+06 | 1.310e+01 | 5.035e+03 | 0.0026 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | 1e+06 | 1.310e+01 | 5.035e+03 | 0.0026 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | 1e+09 | 1.310e+04 | 5.035e+06 | 0.0026 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | 1e+09 | 1.310e+04 | 5.035e+06 | 0.0026 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | 1e+12 | 1.310e+07 | 5.035e+09 | 0.0026 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 25000 | 1e+12 | 1.310e+07 | 5.035e+09 | 0.0026 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | 1e+00 | 3.768e-06 | 1.359e-03 | 0.0028 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | 1e+00 | 3.768e-06 | 1.359e-03 | 0.0028 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | 1e+03 | 3.768e-03 | 1.359e+00 | 0.0028 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | 1e+03 | 3.768e-03 | 1.359e+00 | 0.0028 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | 1e+06 | 3.768e+00 | 1.359e+03 | 0.0028 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | 1e+06 | 3.768e+00 | 1.359e+03 | 0.0028 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | 1e+09 | 3.768e+03 | 1.359e+06 | 0.0028 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | 1e+09 | 3.768e+03 | 1.359e+06 | 0.0028 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | 1e+12 | 3.768e+06 | 1.359e+09 | 0.0028 | soft_clamp | soft_clamp | Y | SAME |
| pilot87 | 6250 | 1e+12 | 3.768e+06 | 1.359e+09 | 0.0028 | soft_clamp | soft_clamp | Y | SAME |

## Part 4: Estimator Error Characterization

| model | iter | cond2 | cond1 | ratio | raw_exact | raw_est | eff_ratio |
|-------|------|-------|-------|-------|-----------|---------|----------|
| pilot4_plain | 0 | 1.251e+05 | 1.268e+06 | 10.138 | 3.690e-05 | 3.741e-04 | 10.138 |
| pilot4_plain | 1500 | 1.354e+05 | 2.306e+06 | 17.036 | 3.994e-05 | 6.804e-04 | 17.036 |
| pilot4_plain | 2250 | 1.355e+05 | 2.231e+06 | 16.463 | 3.999e-05 | 6.584e-04 | 16.463 |
| pilot4_plain | 3000 | 1.257e+05 | 1.667e+06 | 13.259 | 3.709e-05 | 4.917e-04 | 13.259 |
| pilot4_plain | 750 | 1.353e+05 | 1.457e+06 | 10.770 | 3.992e-05 | 4.300e-04 | 10.770 |
| pilot87 | 0 | 2.831e+04 | 1.053e+07 | 372.068 | 5.178e-06 | 1.927e-03 | 372.068 |
| pilot87 | 12500 | 1.133e+05 | 3.244e+07 | 286.408 | 2.072e-05 | 5.934e-03 | 286.408 |
| pilot87 | 18750 | 2.376e+05 | 4.678e+07 | 196.875 | 4.346e-05 | 8.556e-03 | 196.875 |
| pilot87 | 25000 | 7.165e+04 | 2.753e+07 | 384.203 | 1.310e-05 | 5.035e-03 | 384.203 |
| pilot87 | 6250 | 2.060e+04 | 7.432e+06 | 360.791 | 3.768e-06 | 1.359e-03 | 360.791 |

- cond1/cond2: [10.1, 384.2] (mean 166.8)
- eff_ratio: [10.1384, 384.2029] (mean 166.8009)

## Part 5: Acceptance Criterion

- Total points: 210
- SAME: 170 (81.0%)
- MORE conservative: 0 (0.0%)
- LESS conservative: 40 (19.0%)

### Less-conservative decisions

| model | iter | probe | min_xB | exact | est |
|-------|------|-------|--------|-------|-----|
| pilot4_plain | 0 | -1.1*ee | -4.059e-05 | repair | soft_clamp |
| pilot4_plain | 0 | -10*ee | -3.690e-04 | repair | soft_clamp |
| pilot4_plain | 0 | -0.1*es | -3.741e-05 | repair | soft_clamp |
| pilot4_plain | 0 | -es | -3.741e-04 | repair | soft_clamp |
| pilot4_plain | 1500 | -1.1*ee | -4.393e-05 | repair | soft_clamp |
| pilot4_plain | 1500 | -10*ee | -3.994e-04 | repair | soft_clamp |
| pilot4_plain | 1500 | -0.1*es | -6.804e-05 | repair | soft_clamp |
| pilot4_plain | 1500 | -es | -6.804e-04 | repair | soft_clamp |
| pilot4_plain | 2250 | -1.1*ee | -4.399e-05 | repair | soft_clamp |
| pilot4_plain | 2250 | -10*ee | -3.999e-04 | repair | soft_clamp |
| pilot4_plain | 2250 | -0.1*es | -6.584e-05 | repair | soft_clamp |
| pilot4_plain | 2250 | -es | -6.584e-04 | repair | soft_clamp |
| pilot4_plain | 3000 | -1.1*ee | -4.079e-05 | repair | soft_clamp |
| pilot4_plain | 3000 | -10*ee | -3.709e-04 | repair | soft_clamp |
| pilot4_plain | 3000 | -0.1*es | -4.917e-05 | repair | soft_clamp |
| pilot4_plain | 3000 | -es | -4.917e-04 | repair | soft_clamp |
| pilot4_plain | 750 | -1.1*ee | -4.392e-05 | repair | soft_clamp |
| pilot4_plain | 750 | -10*ee | -3.992e-04 | repair | soft_clamp |
| pilot4_plain | 750 | -0.1*es | -4.300e-05 | repair | soft_clamp |
| pilot4_plain | 750 | -es | -4.300e-04 | repair | soft_clamp |
| pilot87 | 0 | -1.1*ee | -5.696e-06 | repair | soft_clamp |
| pilot87 | 0 | -10*ee | -5.178e-05 | repair | soft_clamp |
| pilot87 | 0 | -0.1*es | -1.927e-04 | repair | soft_clamp |
| pilot87 | 0 | -es | -1.927e-03 | repair | soft_clamp |
| pilot87 | 12500 | -1.1*ee | -2.279e-05 | repair | soft_clamp |
| pilot87 | 12500 | -10*ee | -2.072e-04 | repair | soft_clamp |
| pilot87 | 12500 | -0.1*es | -5.934e-04 | repair | soft_clamp |
| pilot87 | 12500 | -es | -5.934e-03 | repair | soft_clamp |
| pilot87 | 18750 | -1.1*ee | -4.780e-05 | repair | soft_clamp |
| pilot87 | 18750 | -10*ee | -4.346e-04 | repair | soft_clamp |
| pilot87 | 18750 | -0.1*es | -8.556e-04 | repair | soft_clamp |
| pilot87 | 18750 | -es | -8.556e-03 | repair | soft_clamp |
| pilot87 | 25000 | -1.1*ee | -1.442e-05 | repair | soft_clamp |
| pilot87 | 25000 | -10*ee | -1.310e-04 | repair | soft_clamp |
| pilot87 | 25000 | -0.1*es | -5.035e-04 | repair | soft_clamp |
| pilot87 | 25000 | -es | -5.035e-03 | repair | soft_clamp |
| pilot87 | 6250 | -1.1*ee | -4.145e-06 | repair | soft_clamp |
| pilot87 | 6250 | -10*ee | -3.768e-05 | repair | soft_clamp |
| pilot87 | 6250 | -0.1*es | -1.359e-04 | repair | soft_clamp |
| pilot87 | 6250 | -es | -1.359e-03 | repair | soft_clamp |

**40 mismatches** out of 210.

## Part 6: Small-Matrix Validation

| name | cond2 | cond1 | ratio | soft | repair |
|------|-------|-------|-------|------|--------|
| diag_1 | 1.000e+00 | 1.000e+00 | 1.000 | Y | Y |
| diag_1e1 | 1.000e+01 | 1.000e+01 | 1.000 | Y | Y |
| diag_1e3 | 1.000e+03 | 1.000e+03 | 1.000 | Y | Y |
| diag_1e6 | 1.000e+06 | 1.000e+06 | 1.000 | Y | Y |
| diag_1e9 | 1.000e+09 | 1.000e+09 | 1.000 | Y | Y |
| diag_1e12 | 1.000e+12 | 1.000e+12 | 1.000 | Y | Y |
| tri_1e1 | 1.253e+00 | 1.818e+00 | 1.451 | Y | Y |
| tri_1e3 | 1.002e+00 | 1.008e+00 | 1.005 | Y | Y |
| tri_1e6 | 1.000e+00 | 1.000e+00 | 1.000 | Y | Y |
| tri_1e9 | 1.000e+00 | 1.000e+00 | 1.000 | Y | Y |
| tri_1e12 | 1.000e+00 | 1.000e+00 | 1.000 | Y | Y |
| rand_s0.1 | 1.204e+00 | 7.458e+01 | 61.960 | Y | Y |
| rand_s1.0 | 7.046e+00 | 5.626e+02 | 79.848 | Y | Y |
| rand_s10.0 | 1.058e+02 | 1.113e+04 | 105.166 | Y | Y |
| rand_s100.0 | 5.320e+02 | 6.780e+04 | 127.448 | Y | Y |
| hilbert_1e4 | 7.809e+22 | 9.178e+23 | 11.753 | Y | Y |
| hilbert_1e6 | 1.212e+25 | 4.581e+26 | 37.809 | Y | Y |
| hilbert_1e8 | 2.222e+27 | 1.982e+28 | 8.918 | Y | Y |
| hilbert_1e10 | 1.378e+29 | 5.389e+30 | 39.104 | Y | Y |
| hilbert_1e12 | 2.100e+31 | 4.107e+32 | 19.553 | Y | Y |
| nearsing_1e-4 | 1.000e+04 | 1.000e+04 | 1.000 | Y | Y |
| nearsing_1e-8 | 1.000e+08 | 1.000e+08 | 1.000 | Y | Y |
| nearsing_1e-12 | 1.000e+12 | 1.000e+12 | 1.000 | Y | Y |

- Soft matches: 23/23
- Repair matches: 23/23

## Summary

- eff_tol ratio: [10.1384, 384.2029]

### Final Safety Assessment

**NOT SAFE YET** -- less-conservative decisions observed.

---
*Generated by `experiment/sparse/stress_condition_gate.py`*
