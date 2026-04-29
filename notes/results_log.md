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
