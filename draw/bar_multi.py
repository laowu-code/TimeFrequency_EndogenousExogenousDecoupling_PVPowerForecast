import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.font_manager import FontProperties
from matplotlib import cm
from matplotlib.colors import Normalize, to_rgba

rc('font', family='Arial')

# Data setup (same as before)
data = {
    "Dataset": ["A"] * 9 + ["B"] * 9 + ["C"] * 9 + ["D"] * 9,
    "Model": ["TCN", "FreTS", "LSTM", "GRU", "Pyraformer", "PatchTST", "iTransformer", "TimeXer", "Proposed"] * 4,
    "MAE": [0.2422, 0.2274, 0.2216, 0.2308, 0.2158, 0.2558, 0.2375, 0.2516, 0.2048,
            0.1732, 0.162, 0.1667, 0.1446, 0.1493, 0.1683, 0.1467, 0.1587, 0.1356,
            0.6897, 0.6323, 0.6959, 0.644, 0.7038, 0.6942, 0.6578, 0.6962, 0.6163,
            0.6754, 0.5963, 0.6173, 0.6335, 0.6413, 0.6769, 0.6482, 0.7199, 0.5888],
    "RMSE": [0.5425, 0.5092, 0.5454, 0.5355, 0.5186, 0.5371, 0.5304, 0.5072, 0.5021,
             0.399, 0.3594, 0.4279, 0.366, 0.3859, 0.3567, 0.3575, 0.3574, 0.3545,
             1.416, 1.4015, 1.4116, 1.3867, 1.4424, 1.3985, 1.4202, 1.3761, 1.3739,
             1.3296, 1.267, 1.2738, 1.2616, 1.323, 1.2807, 1.281, 1.2722, 1.2299],
    r'$\mathrm{R^2}$': [0.9506, 0.9565, 0.9501, 0.9519, 0.9549, 0.9516, 0.9528, 0.9568, 0.9577,
                        0.942, 0.9529, 0.9333, 0.9512, 0.9457, 0.9536, 0.9534, 0.9534, 0.9542,
                        0.9433, 0.9445, 0.9437, 0.9457, 0.9412, 0.9447, 0.943, 0.9465, 0.9467,
                        0.9183, 0.9258, 0.925, 0.9264, 0.9191, 0.9242, 0.9241, 0.9252, 0.9301],
    "MBE": [0.0951, 0.0819, 0.098, 0.1283, -0.0235, 0.0246, 0.0595, 0.0624, 0.0231,
            0.0793, 0.0571, 0.0761, 0.061, 0.0224, 0.0624, 0.0399, 0.0284, 0.0414,
            0.1578, -0.0595, 0.0314, 0.1538, 0.0118, -0.1332, -0.0438, -0.0707, 0.0283,
            0.0908, 0.0768, 0.0339, 0.0381, 0.0499, 0.0983, -0.0017, 0.0159, 0.0942],
}

# Melting DataFrame for plotting
df = pd.DataFrame(data).melt(id_vars=["Dataset", "Model"], var_name="Metric")

# Subplot grid
fig, axes = plt.subplots(4, 4, figsize=(24, 16))
ksize = 25

datasets = ['A', 'B', 'C', 'D']
metrics = ['MAE', 'RMSE', r'$\mathrm{R^2}$', 'MBE']
china_colors = ['#4CAF50', '#2196F3', '#FF5722', '#9C27B0', '#FFC107', '#FF9800', '#00BCD4', '#8BC34A', '#FFEB3B', '#9E9E9E']
bar_width = 0.8

# Main plot loop
for i, dataset in enumerate(datasets):
    for j, metric in enumerate(metrics):
        dataset_metric_data = df[(df['Dataset'] == dataset) & (df['Metric'] == metric)].copy()
        models = list(dataset_metric_data['Model'])[::-1]
        values = list(dataset_metric_data['value'])[::-1]

        # Normalize absolute values for alpha/color strength
        norm = Normalize(vmin=min(map(abs, values)), vmax=max(map(abs, values)))
        base_colors = china_colors[::-1]  # Reverse to match data order

        # Adjust each base color by strength (darken with value)
        colors = []
        for idx, val in enumerate(values):
            base = to_rgba(base_colors[idx % len(base_colors)])
            strength = norm(abs(val))
            # Use RGB * strength to darken
            adjusted = (base[0] * strength, base[1] * strength, base[2] * strength, 1.0)
            colors.append(adjusted)

        ax = axes[i, j]
        x_pos = [j * bar_width + p for p in range(len(models))]

        ax.barh(x_pos, values, bar_width, color=colors, edgecolor='black', linewidth=2)

        if i == 3:
            ax.set_xlabel(f"{metric}", fontsize=ksize)
        if j == 0:
            simhei_font = FontProperties(family='SimHei', size=ksize)
            ax.set_ylabel(f" 数据集 {dataset}", fontsize=ksize, fontproperties=simhei_font)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models, fontsize=ksize)

        x_min, x_max = dataset_metric_data['value'].min(), dataset_metric_data['value'].max()
        margin = 0.1 * (x_max - x_min)
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.tick_params(axis='x', labelsize=ksize)
        ax.tick_params(axis='y', labelsize=ksize)

for ax in axes.flat:
    ax.label_outer()

plt.subplots_adjust(hspace=0.06, wspace=0.2, top=0.9, bottom=0.05, left=0.2, right=0.97)
plt.savefig('../pic/bar_chapt3_gradient_model.svg', format='SVG', dpi=800)
plt.show()
