import matplotlib.pyplot as plt
import numpy as np


# Fixing random state for reproducibility
np.random.seed(42)

grid = np.random.rand(4, 4, 18)

fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

for idx, ax in enumerate(axs.flat):
    ax.imshow(grid[:, :, idx], cmap='viridis')
    ax.set_title(str(idx))

plt.tight_layout()
plt.show()