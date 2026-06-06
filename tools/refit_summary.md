# Sigmoid refit v2: A_0 := 1/C fixed at random-guess floor

Three-parameter fit A(s) = 0.1 + (A_inf - 0.1)/(1+exp(-beta(s - s_0))) with A_0 fixed at 1/C = 0.1 (10-class random guess). v1 (4-parameter) leaves A_0 unconstrained in [-0.05, 1.0] and on most cells pins it to the lower bound, which is unphysical.

## Per-stratum comparison

| dataset | method | n | v1 alpha | v2 alpha | v1 gamma | v2 gamma | v1 R2adj | v2 R2adj |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cifar_pca | magnitude | 351 | -0.133+/-0.011 | -0.119+/-0.007 | +0.390+/-0.017 | +0.279+/-0.011 | 0.872 | 0.898 |
| cifar_pca | random | 351 | -0.099+/-0.014 | -0.084+/-0.015 | +0.037+/-0.019 | +0.119+/-0.020 | 0.304 | 0.387 |
| cifar_pca | wanda | 351 | -0.309+/-0.016 | -0.338+/-0.012 | +0.594+/-0.026 | +0.242+/-0.015 | 0.904 | 0.912 |
| cifar_resnet | magnitude | 351 | -0.268+/-0.015 | -0.248+/-0.014 | +0.840+/-0.028 | +0.778+/-0.025 | 0.937 | 0.934 |
| cifar_resnet | random | 239 | -0.008+/-0.012 | -0.009+/-0.011 | +0.656+/-0.021 | +0.614+/-0.019 | 0.922 | 0.925 |
| cifar_resnet | wanda | 351 | -0.387+/-0.015 | -0.367+/-0.014 | +0.515+/-0.022 | +0.375+/-0.019 | 0.923 | 0.914 |
| mnist28 | magnitude | 351 | -0.270+/-0.016 | -0.259+/-0.016 | +0.996+/-0.032 | +0.922+/-0.031 | 0.942 | 0.934 |
| mnist28 | random | 351 | -0.073+/-0.014 | -0.076+/-0.014 | +0.614+/-0.024 | +0.585+/-0.023 | 0.891 | 0.887 |
| mnist28 | wanda | 351 | -0.414+/-0.014 | -0.418+/-0.014 | +0.654+/-0.022 | +0.531+/-0.021 | 0.948 | 0.940 |
| sklearn | magnitude | 690 | -0.276+/-0.010 | -0.274+/-0.009 | +0.910+/-0.021 | +0.840+/-0.018 | 0.954 | 0.956 |
| sklearn | random | 690 | -0.091+/-0.011 | -0.090+/-0.010 | +0.510+/-0.017 | +0.482+/-0.015 | 0.872 | 0.886 |
| sklearn | wanda | 690 | -0.350+/-0.011 | -0.340+/-0.011 | +0.764+/-0.021 | +0.715+/-0.019 | 0.938 | 0.942 |

## v1 vs v2 1-sigma agreement

- Strata where |alpha_v1 - alpha_v2| <= max(SE): 7 / 12
- Strata where |gamma_v1 - gamma_v2| <= max(SE): 0 / 12
