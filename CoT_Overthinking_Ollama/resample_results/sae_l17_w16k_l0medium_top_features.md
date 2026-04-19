# SAE feature contrast: right->wrong vs right->right

- Model: `google/gemma-3-4b-it`
- SAE: `gemma-scope-2-4b-it-res` / `layer_17_width_16k_l0_medium` (16384 features)
- Hook: `blocks.17.hook_resid_post` (layer 17 residual)
- Position: last prompt token (end of `<start_of_turn>model\n`)
- n(right->wrong) = 108, n(right->right) = 396
- Neuronpedia: https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/<feature>

## Top 30 features higher in right->wrong (over-active when model later flips)

| feature | cohen_d | mean_RW | mean_RR | active_RW | active_RR | link |
|---------|---------|---------|---------|-----------|-----------|------|
| 279 | +0.804 | 1056.581 | 947.684 | 1.00 | 1.00 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/279 |
| 4103 | +0.655 | 9.624 | 0.459 | 0.11 | 0.01 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4103 |
| 589 | +0.625 | 464.890 | 407.121 | 1.00 | 1.00 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/589 |
| 471 | +0.585 | 68.096 | 32.208 | 0.49 | 0.28 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/471 |
| 2072 | +0.572 | 26.402 | 5.171 | 0.17 | 0.03 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2072 |
| 346 | +0.564 | 106.261 | 55.504 | 0.58 | 0.32 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/346 |
| 1659 | +0.519 | 10.093 | 0.602 | 0.07 | 0.01 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1659 |
| 337 | +0.483 | 243.322 | 214.817 | 1.00 | 0.99 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/337 |
| 166 | +0.474 | 212.830 | 173.059 | 0.95 | 0.89 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/166 |
| 6055 | +0.473 | 139.659 | 120.192 | 0.97 | 0.91 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/6055 |
| 4551 | +0.455 | 22.774 | 4.477 | 0.11 | 0.02 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4551 |
| 3871 | +0.422 | 10.666 | 1.813 | 0.08 | 0.02 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3871 |
| 565 | +0.414 | 121.226 | 74.938 | 0.57 | 0.39 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/565 |
| 10089 | +0.405 | 27.576 | 8.199 | 0.14 | 0.04 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/10089 |
| 598 | +0.400 | 93.694 | 68.860 | 0.78 | 0.59 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/598 |
| 10753 | +0.397 | 89.307 | 65.065 | 0.71 | 0.54 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/10753 |
| 36 | +0.369 | 23.093 | 10.254 | 0.21 | 0.10 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/36 |
| 3344 | +0.368 | 8.139 | 0.812 | 0.05 | 0.01 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3344 |
| 12577 | +0.368 | 120.503 | 88.473 | 0.68 | 0.51 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/12577 |
| 15506 | +0.367 | 186.799 | 157.691 | 0.91 | 0.79 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/15506 |
| 694 | +0.360 | 52.725 | 32.596 | 0.47 | 0.29 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/694 |
| 11135 | +0.360 | 185.583 | 167.892 | 0.99 | 0.96 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/11135 |
| 1699 | +0.356 | 56.085 | 37.338 | 0.51 | 0.35 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1699 |
| 10341 | +0.354 | 21.355 | 11.604 | 0.31 | 0.17 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/10341 |
| 1935 | +0.349 | 87.872 | 63.911 | 0.64 | 0.46 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1935 |
| 6609 | +0.323 | 12.496 | 2.933 | 0.06 | 0.02 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/6609 |
| 866 | +0.313 | 81.037 | 58.189 | 0.56 | 0.43 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/866 |
| 502 | +0.302 | 87.961 | 66.705 | 0.64 | 0.52 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/502 |
| 12126 | +0.300 | 66.645 | 40.298 | 0.32 | 0.20 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/12126 |
| 5395 | +0.300 | 70.390 | 46.693 | 0.40 | 0.27 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/5395 |

## Top 30 features higher in right->right (absent when model flips)

| feature | cohen_d | mean_RW | mean_RR | active_RW | active_RR | link |
|---------|---------|---------|---------|-----------|-----------|------|
| 1789 | -0.706 | 58.075 | 122.971 | 0.38 | 0.71 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1789 |
| 280 | -0.522 | 79.147 | 129.165 | 0.56 | 0.76 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/280 |
| 4094 | -0.500 | 68.681 | 116.945 | 0.44 | 0.64 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4094 |
| 10704 | -0.499 | 181.286 | 210.319 | 0.87 | 0.96 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/10704 |
| 5948 | -0.483 | 149.896 | 173.272 | 0.85 | 0.95 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/5948 |
| 10693 | -0.480 | 97.159 | 127.404 | 0.67 | 0.83 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/10693 |
| 5077 | -0.467 | 46.302 | 79.212 | 0.33 | 0.57 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/5077 |
| 1893 | -0.447 | 39.951 | 70.743 | 0.29 | 0.51 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1893 |
| 1105 | -0.431 | 527.611 | 550.147 | 1.00 | 1.00 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1105 |
| 1712 | -0.413 | 322.198 | 336.755 | 1.00 | 1.00 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1712 |
| 225 | -0.413 | 464.247 | 500.310 | 1.00 | 1.00 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/225 |
| 1375 | -0.394 | 74.650 | 98.655 | 0.57 | 0.73 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1375 |
| 13662 | -0.382 | 163.971 | 180.276 | 0.92 | 0.97 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/13662 |
| 2434 | -0.366 | 19.406 | 44.115 | 0.12 | 0.28 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2434 |
| 771 | -0.360 | 4.613 | 20.400 | 0.04 | 0.16 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/771 |
| 4112 | -0.356 | 24.402 | 46.854 | 0.18 | 0.34 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4112 |
| 16239 | -0.344 | 45.389 | 75.385 | 0.26 | 0.42 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/16239 |
| 401 | -0.343 | 93.902 | 118.528 | 0.73 | 0.81 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/401 |
| 3115 | -0.341 | 93.463 | 112.402 | 0.72 | 0.86 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3115 |
| 2591 | -0.323 | 8.831 | 21.885 | 0.08 | 0.21 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2591 |
| 3134 | -0.320 | 33.368 | 55.494 | 0.23 | 0.38 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3134 |
| 4476 | -0.320 | 64.302 | 88.454 | 0.43 | 0.58 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4476 |
| 1902 | -0.315 | 3.445 | 16.110 | 0.03 | 0.12 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1902 |
| 3246 | -0.302 | 104.117 | 119.817 | 0.75 | 0.87 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3246 |
| 34 | -0.296 | 1922.132 | 1973.070 | 1.00 | 1.00 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/34 |
| 11468 | -0.294 | 157.414 | 165.681 | 0.99 | 0.99 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/11468 |
| 153 | -0.289 | 42.532 | 61.285 | 0.35 | 0.52 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/153 |
| 186 | -0.287 | 341.121 | 377.507 | 0.96 | 0.98 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/186 |
| 5 | -0.284 | 100.045 | 117.949 | 0.71 | 0.87 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/5 |
| 11387 | -0.273 | 40.578 | 60.827 | 0.28 | 0.40 | https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/11387 |
