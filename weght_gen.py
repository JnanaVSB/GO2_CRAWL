import numpy as np

ckpt = np.load("/home/jnana/ARLTask/Go2_crawl/results/cmaes_results_newV1_bfs10/checkpoints/checkpoint_gen_0359.npz")
print(f"Gen: {ckpt['gen']}")
print(f"Best reward: {ckpt['best_reward']}")
print(f"Best weights shape: {ckpt['best_weights'].shape}")

np.save("best_weights_from_checkpoint_bfs20.npy", ckpt['best_weights'])
print("Saved best_weights_from_checkpoint.npy")