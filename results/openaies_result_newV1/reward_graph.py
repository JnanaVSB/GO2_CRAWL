import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("training_log.csv")

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(df['gen'], df['best_reward'], color='blue', 
        alpha=0.3, linewidth=0.8)

window = 20
if len(df) > window:
    running_avg = np.convolve(df['best_reward'], 
                              np.ones(window)/window, 
                              mode='valid')
    ax.plot(df['gen'].values[window-1:], running_avg, 
            color='blue', linewidth=2, label='Running average')

ax.set_xlabel('Generation')
ax.set_ylabel('Best Reward')
ax.set_title('OpenAI-ES Optimization Progress')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('openaies_reward_history.png', dpi=150)
print("Saved")