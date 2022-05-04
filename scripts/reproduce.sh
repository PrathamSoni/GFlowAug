conda activate pytorch_latest_p37
pip install -r requirements.txt
python setup.py install
python examples/run_expt.py --dataset iwildcam --algorithm deepCORAL --root_dir data --log_dir ./logs/iwildcam --download true
python examples/run_expt.py --dataset globalwheat --algorithm ERM --root_dir data --log_dir ./logs/globalwheat --download true
python examples/run_expt.py --dataset poverty --algorithm ERM --root_dir data --log_dir ./logs/poverty --download true
