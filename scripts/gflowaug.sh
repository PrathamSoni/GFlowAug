conda activate pytorch_latest_p37
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install torch-geometric
python setup.py install
python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --root_dir data --log_dir ./logs/iwildcam_pretrain --download true --additional_train_transform gflowaug
python examples/evaluate.py ./logs/iwildcam_pretrain ./results --dataset iwildcam --root-dir data --suppress_replicates True

python examples/run_expt.py --dataset globalwheat --algorithm ERM --root_dir data --log_dir ./logs/globalwheat_pretrain --download true --additional_train_transform gflowaug
python examples/evaluate.py ./logs/globalwheat_pretrain ./results --dataset globalwheat --root-dir data --suppress_replicates True

python examples/run_expt.py --dataset poverty --algorithm ERM --root_dir data --log_dir ./logs/poverty_pretrain --download true --additional_train_transform gflowaug
python examples/evaluate.py ./logs/poverty_pretrain ./results --dataset poverty --root-dir data --suppress_replicates True
