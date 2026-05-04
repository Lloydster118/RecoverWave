# Results log

## 24 Apr - first real CV run

Quick sanity ridge fit on real data:
- Train: 5-fold chronological
- MAE on persistence baseline: 27.6 (worse than expected, big swings)
- Ridge: 20.1
- GBM: 19.9 (Pearson r ~0.25)

Music features are weak. Sleep features dominate.

## 29 Apr
Tried seq_len=14 (two weeks) - made it worse. Reverted to 7. Attention naturally concentrates on last 3 days so longer context is just noise.

## 4 May - final CV numbers

5-fold chronological CV, 159 test days each, seed=20260504:

| Model         | MAE          | Pearson r |
|---------------|--------------|-----------|
| SeasonalDoW   | 19.75 ± 1.36 | -         |
| GBM           | 19.93 ± 2.33 | 0.25      |
| Ridge         | 19.97 ± 2.67 | 0.21      |
| LSTM+Attn     | 20.20 ± 2.25 | -0.02     |
| Persistence   | 27.67 ± 4.23 | -0.05     |

Attention weights: t=0.353, t-1=0.338, t-2=0.253 (~94% on 3 most recent days)

Mediation: a=-0.0222 hours/track, b=+4.94 recovery/hour, indirect ~25% of total effect.

Writing up.
