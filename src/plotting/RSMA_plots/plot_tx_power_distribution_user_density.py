import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.ticker as mticker
from pathlib import (
    Path,
)
from src.config.config import (
    Config,
)
from src.config.config_plotting import (
    PlotConfig,
    save_figures,
    generic_styling, change_lightness,
)


def plot_tx_power_distribution_stacked_bars(
        paths,
        name,
        width,
        height,
        plots_parent_path,
        legend: list | None = None,          # labels für die 4 Approaches (x-axis)
        approach_colors: list | None = None, # optional: bar-edge/outline pro approach
        user_colors: list | None = None,     # Farben für die User-Segmente
        metric_key: str = 'tx_power_per_user',
        distance_idx: int = 0,               # du hast aktuell nur einen distance sweep punkt -> 0
        show_std: bool = False,              # optional: errorbar auf Gesamtleistung
) -> None:
    """
    Erwartet pro gzip: pickle.dump([distance_sweep_range, metrics], ...)
    wobei metrics[metric_key]['mean'] shape (n_dist, user_nr) ist.
    Plottet pro Ansatz einen gestapelten Balken (User als Segmente).
    """

    # load all
    data = []
    for p in paths:
        with gzip.open(p, 'rb') as f:
            data.append(pickle.load(f))  # [distance_sweep_range, metrics]

    n_approaches = len(data)

    # infer user_nr from first file
    m0 = data[0][1][metric_key]
    user_nr = m0['mean'].shape[1]

    if legend is None:
        legend = [f'Approach {i}' for i in range(n_approaches)]

    if user_colors is None:
        # fallback colormap
        cmap = plt.get_cmap('tab10')
        user_colors = [cmap(i) for i in range(user_nr)]

    if approach_colors is None:
        approach_colors = [None] * n_approaches

    fig, ax = plt.subplots(figsize=(width, height))

    x = np.arange(n_approaches)
    bar_width = 0.75

    # if user_nr <= 3:
    #     user_hatches = ['///', '...', '\\\\\\']
    # else:
    #     user_hatches = ['///', '\\\\\\', '...', 'xxx', '++', 'oo', '--', '**']

    user_hatches= ['']

    # plot bars
    for a in range(n_approaches):
        metrics = data[a][1]
        mean_mat = np.asarray(metrics[metric_key]['mean'])
        std_mat = np.asarray(metrics[metric_key]['std'])

        powers = mean_mat[distance_idx, :]          # (user_nr,)
        stds = std_mat[distance_idx, :]             # (user_nr,)

        # bottom = 0.0
        # for u in range(user_nr):
        #     ax.bar(
        #         x[a],
        #         powers[u],
        #         width=bar_width,
        #         bottom=bottom,
        #         # color=user_colors[u],
        #         color='white',
        #         hatch=user_hatches[u % len(user_hatches)],
        #         # edgecolor=approach_colors[a],
        #         edgecolor=(user_colors[u] if user_colors[u] is not None else 'black'),
        #         linewidth=1.2,#(1.2 if approach_colors[a] is not None else 0.0),
        #     )
        #     bottom += powers[u]

        bottom = 0.0
        for u in range(user_nr):
            # Zeichne Balken
            ax.bar(
                x[a],
                powers[u],
                width=bar_width,
                bottom=bottom,
                color='white',
                hatch=user_hatches[u % len(user_hatches)],
                edgecolor='black',#(user_colors[u] if user_colors[u] is not None else 'black'),
                linewidth=1.2,
            )

            # ✅ Füge Text hinzu: k = u+1 in der Mitte des Balkensegments
            center_y = bottom + powers[u] / 2
            ax.text(
                x[a], center_y,
                f'$k={u+1}$',
                ha='center', va='center',
                fontsize=8,  # passt an die Größe an
                fontweight='bold',
                color='black',
                # Optional: Rahmen um Text
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none")
            )

            bottom += powers[u]

        if show_std:
            # std der Summe (angenommen unabhängig): sqrt(sum(std_u^2))
            total_std = float(np.sqrt(np.sum(stds ** 2)))
            total_mean = float(np.sum(powers))
            ax.errorbar(
                x[a], total_mean,
                yerr=total_std,
                color='k',
                capsize=3,
                linewidth=1.0,
                fmt='none',
                zorder=10,
            )

    # styling / labels


    ax.set_xticks(x)
    ax.set_xticklabels(legend, rotation=0)
    ax.set_ylabel(r'Transmit Power $P$ [W]')
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    # legend for user colors
    # user_patches = [mpatches.Patch(color=user_colors[u], label=f'User {u}') for u in range(user_nr)]
    # ax.legend(handles=user_patches, ncols=min(user_nr, 4), loc='best')

    user_patches = [
        mpatches.Patch(facecolor='white', edgecolor=(user_colors[u] if user_colors[u] is not None else 'black'),
                       hatch=user_hatches[u % len(user_hatches)],
                       label=f'User $k={u+1}$')
        for u in range(user_nr)
    ]
    # ax.legend(handles=user_patches, ncols=min(user_nr, 4), loc='lower right')


    generic_styling(ax=ax)
    # ax.legend(
    #     handles=user_patches,
    #     ncols=min(user_nr, 4),
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.15),  # x=zentriert, y=unterhalb der Achse
    #     frameon=True,
    #     borderaxespad=0.0
    # )

    fig.tight_layout(pad=0)
    #
    # # Platz unten für die Legende schaffen
    # fig.subplots_adjust(bottom=0.22)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name, padding=0.05)

if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error', '03_power_distribution',
             'testing_mmse_usersweep_tx_500_50000.gzip'),
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error','03_power_distribution',
             'testing_learned_usersweep_tx_500_50000.gzip'),
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error', '03_power_distribution',
             'testing_learned_rsma_full_usersweep_tx_500_50000.gzip'),
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error', '03_power_distribution',
             'testing_learned_rsma_power_common_usersweep_tx_500_50000.gzip'),
    ]



    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.99

    plot_legend = [
        r'RSMA $\alpha*$',
        r'L-SDMA',
        r'L-RSMA',
        r'H-RSMA',
    ]
    # plot_markerstyle = [ 'o', 's', 'd', 'x']
    # plot_colors = [ change_lightness(plot_cfg.cp2['black'], 1), plot_cfg.cp3['blue2'], change_lightness(plot_cfg.cp3['red2'], 1), plot_cfg.cp3['red1']]
    # plot_linestyles = [ '-', '-', '-', '-',]

    plot_tx_power_distribution_stacked_bars(
        paths=data_paths,  # deine 4 gzip files
        name='tx_power_dist_one_point',
        width=plot_width,
        height=plot_height,
        legend=[r'RSMA $\alpha*$', r'L-SDMA', r'L-RSMA', r'H-RSMA'],
        # user_colors=[plot_cfg.cp3['blue2'], plot_cfg.cp3['red2'], plot_cfg.cp3['black']],  # z.B. 3 user
        plots_parent_path=plot_cfg.plots_parent_path,
        distance_idx=0,
        show_std=False,
    )
    plt.show()
