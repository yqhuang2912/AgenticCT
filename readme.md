# 生成仿真数据
```bash
cd /data/hyq/codes/AgenticCT/src/dataset
conda activate py312_torch271_cu128
python python generate_deeplesion_dataset.py --output_dir /data/hyq/codes/AgenticCT/data/deeplesion/lact --degradation_type lact
```

# 从生成的仿真数据 dataset_info.csv生成QE Dataset Info
```bash
cd /data/hyq/codes/AgenticCT/src/dataset
conda activate py312_torch271_cu128
python generate_qe_dataset_info.py \
  --raw_dataset_info_files \
    /data/hyq/codes/AgenticCT/data/deeplesion/ldct/dataset_info.csv \
    /data/hyq/codes/AgenticCT/data/deeplesion/lact/dataset_info.csv \
    /data/hyq/codes/AgenticCT/data/deeplesion/svct/dataset_info.csv \
  --qe_dataset_info_out /data/hyq/codes/AgenticCT/data/qe/deeplesion/dataset_info_qe.csv \
  --dedup_clean
```

# 划分训练测试集
```bash
cd /data/hyq/codes/AgenticCT/src/dataset
conda activate py312_torch271_cu128
python train_test_split.py --input_csv /data/hyq/codes/AgenticCT/data/deeplesion/lact/dataset_info.csv --ratio 0.75
```