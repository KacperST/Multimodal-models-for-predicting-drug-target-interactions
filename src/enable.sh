# K2 (scaffold, domyślny)
uv run run_multiseed.py configs/lincs/ --split-strategy scaffold

# K3 (niewidziane białka)
uv run run_multiseed.py configs/lincs/ --split-strategy cold_target

# K4 (niewidziane leki I białka)
uv run run_multiseed.py configs/lincs/ --split-strategy cold_both
