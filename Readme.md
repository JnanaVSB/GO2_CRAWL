# Go2 Quadruped Locomotion (PCA + DMP + CMA-ES)

Adapted from [humanoid-spider-walk](https://github.com/k-pratyush/humanoid-spider-walk). Gets the Unitree Go2 to crawl forward in MuJoCo using PCA for dimensionality reduction, rhythmic DMPs for trajectory generation, and CMA-ES for optimization.

Best result: **44.9 cm forward** with 20 basis functions over 800 generations.

## Demo

| 10 bfs (23.0 cm) | 15 bfs (25.7 cm) | 20 bfs (44.9 cm) |
|---|---|---|
| ![10bfs](videos/go2_locomotion_bfs10.gif) | ![15bfs](videos/go2_locomotion_bfs15.gif) | ![20bfs](videos/go2_locomotion_bfs20.gif) |

## How to run
```bash
pip install numpy pandas scikit-learn mujoco cma matplotlib imageio

# 1. Build initial DMP weights
python pca_dmp_init.py

# 2. Train
python optimize/train_cmaes.py

# 3. Evaluate
python test/evaluate_best.py
```

## Results

| bfs | Parameters | Distance |
|-----|-----------|----------|
| 10  | 20        | 23.0 cm  |
| 15  | 30        | 25.7 cm  |
| 20  | 40        | 44.9 cm  |

OpenAI-ES was also compared but produced no forward motion after 800 generations.

Videos and training logs: [Google Drive](https://drive.google.com/drive/folders/1SqeYJS50ECIdGQgQE2sHkJiJAsyYAuGn?usp=sharing)

## Report

Full development report with detailed analysis: [Report PDF](link-to-report)