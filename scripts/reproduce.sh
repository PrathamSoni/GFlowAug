conda activate pytorch_latest_p37
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install torch-geometric
python setup.py install
python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --root_dir data --log_dir ./logs/iwildcam --download true
python examples/run_expt.py --dataset globalwheat --algorithm ERM --root_dir data --log_dir ./logs/globalwheat --download true
python examples/run_expt.py --dataset poverty --algorithm ERM --root_dir data --log_dir ./logs/poverty --download true
python examples/evaluate.py ./logs/iwildcam ./results --dataset iwildcam --root-dir data --suppress_replicates True
python examples/evaluate.py ./logs/globalwheat ./results --dataset globalwheat --root-dir data --suppress_replicates True
python examples/evaluate.py ./logs/poverty ./results --dataset poverty --root-dir data --suppress_replicates True
