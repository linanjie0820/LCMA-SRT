# inspect_supervisions.py
import torch, os, json
batch_path = "/pfs/asr/ASR_AND_AST/asr2_and_ast3/exp/asr3_and_ast3_stage2_moe_tgtLangid_temp1.5_entropy0.5_20251208/batch-bdd640fb-0667-1ad1-1c80-317fa3b1799d.pt"
assert os.path.exists(batch_path), batch_path
data = torch.load(batch_path, map_location="cpu")

print("Keys:", list(data.keys()))
inputs = data["inputs"]  # tensor (B, T, feat)
superv = data["supervisions"]

print("inputs shape:", inputs.shape)
print("supervisions type:", type(superv))

# If supervisions is a dict, print its keys and per-sample entries
if isinstance(superv, dict):
    print("supervisions keys:", list(superv.keys()))
    # typical structure: superv = {"id": [...], "text": [...], "start": [...], "duration": [...], ...}
    for k in superv.keys():
        v = superv[k]
        print(f"- {k}: type={type(v)}; len={len(v) if hasattr(v, '__len__') else 'NA'}")
    # Print first few samples metadata
    print("\nFirst 5 supervision entries:")
    B = inputs.shape[0]
    for i in range(min(5, B)):
        print(f"=== sample {i} ===")
        for k in superv.keys():
            try:
                print(f"{k}: {superv[k][i]}")
            except Exception:
                print(f"{k}: <cannot index>")
else:
    print("supervisions is not a dict, repr:")
    print(repr(superv))
