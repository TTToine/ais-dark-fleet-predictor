import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import average_precision_score, roc_auc_score

print("=" * 70)
print("D1 - Train vs Test distribution")
print("=" * 70)

df = pd.read_parquet('data/processed/ais_enriched.parquet')
with open('models/holdout_mmsis.json') as f:
    holdout = json.load(f)
test_mmsis = set(holdout['holdout_mmsis'])
print(f"Holdout mmsis loaded: {len(test_mmsis)}")

df['split'] = df['MMSI'].apply(lambda m: 'test' if m in test_mmsis else 'train')

for split in ['train', 'test']:
    sub = df[df['split'] == split]
    pos = sub[sub['target_dark_fleet'] == 1]
    neg = sub[sub['target_dark_fleet'] == 0]
    print(f"--- {split} ---")
    print(f"  Total: {len(sub)}, positives: {len(pos)} ({100*len(pos)/len(sub):.4f}%)")
    print(f"  SOG pos: mean={pos['SOG'].mean():.2f} std={pos['SOG'].std():.2f}")
    print(f"  SOG neg: mean={neg['SOG'].mean():.2f} std={neg['SOG'].std():.2f}")
    print(f"  Unique vessels: {sub['MMSI'].nunique()}")
    print(f"  Vessels with at least 1 positive: {pos['MMSI'].nunique()}")

print()
print("=" * 70)
print("D2 - Model tree structure")
print("=" * 70)

with open('models/lgb_dark_fleet.pkl', 'rb') as f:
    m = pickle.load(f)
print(f"Num trees: {m.num_trees()}")
print(f"Feature names: {m.feature_name()}")

tree_info = m.dump_model()['tree_info']
print(f"Trees in dump: {len(tree_info)}")

def walk(node, depth=0):
    indent = '  ' * (depth + 2)
    if 'leaf_index' in node:
        print(f"{indent}LEAF value={node['leaf_value']:.4f}")
    else:
        print(f"{indent}SPLIT feat_idx={node['split_feature']} thresh<={node['threshold']:.4f} gain={node['split_gain']:.2f}")
        walk(node['left_child'], depth + 1)
        walk(node['right_child'], depth + 1)

for i, t in enumerate(tree_info[:2]):
    print(f"--- Tree {i} ---")
    print(f"  Num leaves: {t['num_leaves']}")
    walk(t['tree_structure'])

print()
print("=" * 70)
print("D3 - Stupid predictor baselines on test")
print("=" * 70)

test = df[df['split'] == 'test'].copy()
y_test = test['target_dark_fleet'].values

score_sog_inv = 1.0 / (test['SOG'].values + 1)
pr_sog = average_precision_score(y_test, score_sog_inv)
roc_sog = roc_auc_score(y_test, score_sog_inv)

print(f"Test rows: {len(test)}, positives: {int(y_test.sum())}, baseline: {100*y_test.mean():.4f}%")
print(f"Stupid (1/(SOG+1)): PR-AUC={pr_sog:.4f}, ROC-AUC={roc_sog:.4f}")

score_thresh = (test['SOG'] < 6.0).astype(float).values
pr_thresh = average_precision_score(y_test, score_thresh)
roc_thresh = roc_auc_score(y_test, score_thresh)
print(f"Threshold SOG<6: PR-AUC={pr_thresh:.4f}, ROC-AUC={roc_thresh:.4f}")

score_thresh4 = (test['SOG'] < 4.0).astype(float).values
pr_thresh4 = average_precision_score(y_test, score_thresh4)
roc_thresh4 = roc_auc_score(y_test, score_thresh4)
print(f"Threshold SOG<4: PR-AUC={pr_thresh4:.4f}, ROC-AUC={roc_thresh4:.4f}")