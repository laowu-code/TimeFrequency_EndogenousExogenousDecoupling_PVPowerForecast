import numpy as np
from matplotlib import pyplot as plt

feature_map_GTU = np.load('../data/feature_map_GTU.npy')
feature_map_CA = np.load('../data/feature_map_CA.npy')
ksize = 20

fig, axes = plt.subplots(1, 2, figsize=(20, 6))

for ax, feature_map, title in zip(
    axes,
    [feature_map_CA, feature_map_GTU],
    ["Feature map of CA", "Feature map of GTU"]
):
    im = ax.imshow(feature_map.reshape(10, -1), aspect=6, cmap="viridis", interpolation="none")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Values', fontsize=ksize)
    cbar.ax.tick_params(labelsize=ksize)
    ax.set_xlabel("Index", fontsize=ksize)
    ax.set_title(title, fontsize=ksize)
    ax.set_yticks([0, 5, 9])
    ax.set_yticklabels(['1', '6', '10'])
    ax.tick_params(axis='x', labelsize=ksize)
    ax.tick_params(axis='y', labelsize=ksize)

plt.subplots_adjust(wspace=-0.1)
plt.savefig('../pic/feature_map_CA_GTU.svg', format='SVG', dpi=800)

plt.show()
