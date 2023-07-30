eval "$(conda shell.bash hook)"

conda activate kit-sch-ge-2021-cell_segmentation_ve

# batch size adjusted for 2 GPUs with 24GB VRAM each

# Evaluation of the specialized models trained on GTs+STs from the reproduced kit-sch-ge training dataset (for multiple thresholds, the option --fuse_z_seeds has not been applied in evaluation but in inference for some 3D cell types)
python ./eval.py --cell_type "BF-C2DL-HSC" --mode "GT" --model "BF-C2DL-HSC_GT+ST_kit-sch-ge" --batch_size 16 --artifact_correction --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "BF-C2DL-MuSC" --mode "GT" --model "BF-C2DL-MuSC_GT+ST_kit-sch-ge" --batch_size 8 --artifact_correction --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "DIC-C2DH-HeLa" --mode "GT" --model "DIC-C2DH-HeLa_GT+ST_kit-sch-ge" --batch_size 16 --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "Fluo-C2DL-MSC" --mode "GT" --model "Fluo-C2DL-MSC_GT+ST_kit-sch-ge" --batch_size 16 --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "Fluo-C3DH-A549" --mode "GT" --model "Fluo-C3DH-A549_GT+ST_kit-sch-ge" --batch_size 8 --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "Fluo-C3DH-H157" --mode "GT" --model "Fluo-C3DH-H157_GT+ST_kit-sch-ge" --batch_size 8 --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "Fluo-C3DL-MDA231" --mode "GT" --model "Fluo-C3DL-MDA231_GT+ST_kit-sch-ge" --batch_size 8 --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "Fluo-N2DH-GOWT1" --mode "GT" --model "Fluo-N2DH-GOWT1_GT+ST_kit-sch-ge" --batch_size 16 --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "Fluo-N3DH-CE" --mode "GT" --model "Fluo-N3DH-CE_GT+ST_kit-sch-ge" --batch_size 8 --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "Fluo-N2DL-HeLa" --mode "GT" --model "Fluo-N2DL-HeLa_GT+ST_kit-sch-ge" --batch_size 16 --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "Fluo-N3DH-CHO" --mode "GT" --model "Fluo-N3DH-CHO_GT+ST_kit-sch-ge" --batch_size 8 --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "PhC-C2DH-U373" --mode "GT" --model "PhC-C2DH-U373_GT+ST_kit-sch-ge" --batch_size 16 --th_cell 0.07 --th_seed 0.35 0.45
python ./eval.py --cell_type "PhC-C2DL-PSC" --mode "GT" --model "PhC-C2DL-PSC_GT+ST_kit-sch-ge" --batch_size 16 --th_cell 0.07 --th_seed 0.35 0.45

conda deactivate

