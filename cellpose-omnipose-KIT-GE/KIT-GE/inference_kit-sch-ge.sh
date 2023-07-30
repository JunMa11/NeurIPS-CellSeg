eval "$(conda shell.bash hook)"

conda activate kit-sch-ge-2021-cell_segmentation_ve

# Predictions of the challenge datasets with the specialized models trained on GTs and STb from the reproduced kit-sch-ge training dataset
python ./infer.py --cell_type "BF-C2DL-HSC" --model "kit-sch-ge/BF-C2DL-HSC_GT+ST_model" --batch_size 16 --th_cell 0.07 --th_seed 0.35 --artifact_correction
python ./infer.py --cell_type "BF-C2DL-MuSC" --model "kit-sch-ge/BF-C2DL-MuSC_GT+ST_model" --batch_size 8 --th_cell 0.07 --th_seed 0.35 --artifact_correction
python ./infer.py --cell_type "DIC-C2DH-HeLa" --model "kit-sch-ge/DIC-C2DH-HeLa_GT+ST_model" --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./infer.py --cell_type "Fluo-C2DL-MSC" --model "kit-sch-ge/Fluo-C2DL-MSC_GT+ST_model"--batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./infer.py --cell_type "Fluo-C3DH-A549" --model "kit-sch-ge/Fluo-C3DH-A549_GT+ST_model"--batch_size 8 --th_cell 0.07 --th_seed 0.35 --fuse_z_seeds
python ./infer.py --cell_type "Fluo-C3DH-H157" --model "kit-sch-ge/Fluo-C3DH-H157_GT+ST_model" --batch_size 8 --th_cell 0.07 --th_seed 0.45 --fuse_z_seeds
python ./infer.py --cell_type "Fluo-C3DL-MDA231" --model "kit-sch-ge/Fluo-C3DL-MDA231_GT+ST_model"--batch_size 8 --th_cell 0.07 --th_seed 0.45 --fuse_z_seeds
python ./infer.py --cell_type "Fluo-N2DH-GOWT1" --model "kit-sch-ge/Fluo-N2DH-GOWT1_GT+ST_model"--batch_size 16 --th_cell 0.07 --th_seed 0.45
python ./infer.py --cell_type "Fluo-N2DL-HeLa" --model "kit-sch-ge/Fluo-N2DL-HeLa_GT+ST_model" --batch_size 16 --th_cell 0.07 --th_seed 0.35
python ./infer.py --cell_type "Fluo-N3DH-CE" --model "kit-sch-ge/Fluo-N3DH-CE_GT+ST_model" --batch_size 8 --th_cell 0.07 --th_seed 0.45
python ./infer.py --cell_type "Fluo-N3DH-CHO" --model "kit-sch-ge/Fluo-N3DH-CHO_GT+ST_model" --batch_size 8 --th_cell 0.07 --th_seed 0.45  --fuse_z_seeds
python ./infer.py --cell_type "PhC-C2DH-U373" --model "kit-sch-ge/PhC-C2DH-U373_GT+ST_model"--batch_size 8 --th_cell 0.07 --th_seed 0.35
python ./infer.py --cell_type "PhC-C2DL-PSC" --model "kit-sch-ge/PhC-C2DL-PSC_GT+ST_model"--batch_size 16 --th_cell 0.07 --th_seed 0.45

conda deactivate

