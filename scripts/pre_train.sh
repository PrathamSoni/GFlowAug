conda activate pytorch_latest_p37
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install torch-geometric
python setup.py install

python examples/main.py --dataset iwildcam  --root_dir data --log_dir ./logs/iwildcam_pretrain --download true --algorithm ERM --n_epochs 1
python examples/main.py --dataset poverty  --root_dir data --log_dir ./logs/poverty_pretrain --download true --algorithm ERM --n_epochs 1
python examples/main.py --dataset globalwheat  --root_dir data --log_dir ./logs/globalwheat_pretrain --download true --algorithm ERM --n_epochs 1
