Gravitational Waves Glitch Classification
===
A pytorch implementation of GW glitches classification reproducing the results of the following paper:

> Razzano, M., & Cuoco, E. (2018). Image-based deep learning for classification of noise transients in gravitational wave detectors. Classical and Quantum Gravity, 35(9), 095016.

## Results on Test subset
|||
|---------:| ------: |
| LogLoss  | 0.00019 |
| Accuracy | 0.99897 |

### Confusion Matrix

|      t/p      | CHIRPLIKE | GAUSS | NOISE |  RD | SCATTEREDLIKE |  SG | WHISTLELIKE |
|--------------:|----------:|------:|------:|----:|--------------:|----:|------------:| 
|     CHIRPLIKE |       279 |       |       |     |               |     |             | 
|         GAUSS |           |   276 |       |     |               | 2   |             | 
|         NOISE |           |       | 280   |     |               |     |             | 
|            RD |           |       |       | 279 |               |     |             | 
| SCATTEREDLIKE |           |       |       |     |           279 |     |             | 
|            SG |           |       |       |     |               | 280 |             | 
|   WHISTLELIKE |           |       |       |     |               |     |         280 | 

### Classification Metrics

|              | precision | recall | f1-score | support |
|-------------:|----------:|-------:|---------:|--------:|
|    CHIRPLIKE |      1.00 |   1.00 |     1.00 |     279 |
|        GAUSS |      1.00 |   0.99 |     1.00 |     278 |
|        NOISE |      1.00 |   1.00 |     1.00 |     280 |
|           RD |      1.00 |   1.00 |     1.00 |     279 |
|SCATTEREDLIKE |      1.00 |   1.00 |     1.00 |     279 |
|           SG |      0.99 |   1.00 |     1.00 |     280 |
|  WHISTLELIKE |      1.00 |   1.00 |     1.00 |     280 |
|                                                        |
|    micro avg |      1.00 |   1.00 |     1.00 |     955 |
|    macro avg |      1.00 |   1.00 |     1.00 |     955 |
| weighted avg |      1.00 |   1.00 |     1.00 |     955 |


## Requirements

- pytorch 0.4.1 + torchvision

## Steps to reproduce

- [Download the dataset](data/)
- Train the model and evaluate the model:
```bash
python main.py
python main.py --evaluate
```
