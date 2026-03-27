#!/usr/bin/env python3
"""Remote check script - upload and execute on server."""
import sys
sys.path.insert(0, "/root/autodl-tmp/BRA_Project")

print("=== Import Check ===")
try:
    from bra_operator_multi import create_bra_operator, BRAConfig, detect_adapter
    print("[OK] bra_operator_multi imports successfully")
except Exception as e:
    print(f"[FAIL] bra_operator_multi: {e}")

try:
    from smoke_test_matrix import test_datasets, MODEL_CONFIGS, DATASET_CONFIGS
    print("[OK] smoke_test_matrix imports successfully")
    print(f"  Models: {list(MODEL_CONFIGS.keys())}")
    print(f"  Datasets: {list(DATASET_CONFIGS.keys())}")
except Exception as e:
    print(f"[FAIL] smoke_test_matrix: {e}")

print("\n=== Dataset Completeness Check ===")
try:
    from smoke_test_matrix import test_datasets, note_minigpt4, RESULTS, print_summary
    test_datasets()
    note_minigpt4()
    print_summary()
except Exception as e:
    print(f"[FAIL] Dataset check error: {e}")
    import traceback
    traceback.print_exc()
