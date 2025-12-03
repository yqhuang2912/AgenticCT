
export CUDA_HOME=/data/hyq/envs/miniconda3/envs/py312_torch271_cu128
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

export MPLCONFIGDIR=/data/hyq/matplotlib_cache

/data/hyq/envs/miniconda3/envs/py312_torch271_cu128/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"


# /data/hyq/envs/miniconda3/envs/py312_torch271_cu128/bin/python lact_fbpconvnet_trainer.py --severity high
# /data/hyq/envs/miniconda3/envs/py312_torch271_cu128/bin/python lact_fbpconvnet_trainer.py --severity medium
# /data/hyq/envs/miniconda3/envs/py312_torch271_cu128/bin/python lact_fbpconvnet_trainer.py --severity low

# /data/hyq/envs/miniconda3/envs/py312_torch271_cu128/bin/python ldct_redcnn_trainer.py --severity high
# /data/hyq/envs/miniconda3/envs/py312_torch271_cu128/bin/python ldct_redcnn_trainer.py --severity medium
# /data/hyq/envs/miniconda3/envs/py312_torch271_cu128/bin/python ldct_redcnn_trainer.py --severity low

# /data/hyq/envs/miniconda3/envs/py312_torch271_cu128/bin/python svct_fbpconvnet_trainer.py --severity high
/data/hyq/envs/miniconda3/envs/py312_torch271_cu128/bin/python svct_fbpconvnet_trainer.py --severity medium
# /data/hyq/envs/miniconda3/envs/py312_torch271_cu128/bin/python svct_fbpconvnet_trainer.py --severity low