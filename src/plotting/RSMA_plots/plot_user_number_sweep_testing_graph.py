import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
from fractions import Fraction

from src.config.config import Config
from src.config.config_plotting import PlotConfig, save_figures, generic_styling, change_lightness


def plot_user_number_sweep_testing_graph(
        paths,
        name,
        width,
        height,
        plots_parent_path,
        legend: list = None,
        colors: list = None,
        markerstyle: list = None,
        linestyles: list = None,
        metric: str = 'sumrate'
) -> None:
    """
    Plotted einen User-Number-Sweep mit konfigurierbarem Styling.
    Kombiniert die Datenlogik des ersten Skripts mit dem Achsen-Formatting des zweiten.
    """
    def get_metric_key(data_dict, match_string):
        for key in data_dict[1].keys():
            if match_string in str(key):
                return key
        raise ValueError(f'Match string "{match_string}" nicht in data_dict[1] gefunden.')

    fig, ax = plt.subplots(figsize=(width, height))

    # Metrik-Schlüssel bestimmen
    if metric == 'sumrate':
        match_string = 'calc_sum_rate'
    elif metric == 'fairness':
        match_string = 'calc_jain_fairness'
    else:
        raise ValueError(f'Unbekannte Metrik: {metric}')

    # Daten laden
    data = []
    for path in paths:
        with gzip.open(path, 'rb') as file:
            data.append(pickle.load(file))

    # Plotten (nur die Kurven, NICHT die Achsen/Legende!)
    for data_id, data_entry in enumerate(data):
        metric_key = get_metric_key(data_entry, match_string)

        marker = markerstyle[data_id] if markerstyle else None
        color = colors[data_id] if colors else None
        linestyle = linestyles[data_id] if linestyles else None


        filled = (data_id == 3)
        step = 3
        offsets = [2, 2, 1, 0]  # pro Kurve ein anderer Start-Offset

        offset = offsets[data_id] % step
        markevery = (offset, step)

        ax.plot(data_entry[0],
                data_entry[1][metric_key]['mean'],
                color=color,
                marker=marker,
                markevery=markevery,
                linestyle=linestyle,
                markerfacecolor=(color if filled else 'none'),)

    # --- Achsen & Formatting (außerhalb der Schleife!) ---
    ax.set_xlabel('User Number $ K $')

    if metric == 'sumrate':
        ax.set_ylabel('Achievable Rate $ R $ [bps/Hz]')
        ax.set_ylim(0, 5.1)  # Vom zweiten Skript übernommen, ggf. anpassen
    elif metric == 'fairness':
        ax.set_ylabel('Fairness $F$')
        ax.set_ylim(0.3, 1.1)
        ax.set_yticks([1 / 3, 2 / 3, 1.0])
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, pos: str(Fraction(y).limit_denominator()))
        )

    if legend:
        ax.legend(legend, ncols=2, loc='lower right')

    # Styling & Speichern
    generic_styling(ax=ax)
    fig.tight_layout(pad=0)
    save_figures(plots_parent_path=plots_parent_path, plot_name=name + '_' + metric, padding=0.05)


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path, '03_user_density',
             # '25km',
             '100km',
             'testing_rsma_genie_sweep_1_12.gzip'),
        Path(cfg.output_metrics_path, '03_user_density',
             # '25km',
             '100km',
             'testing_learned_sac_sweep_1_12.gzip'),
        Path(cfg.output_metrics_path, '03_user_density',
             # '25km',
             '100km',
             'testing_learned_rsma_complete_sweep_1_12.gzip'),
        Path(cfg.output_metrics_path, '03_user_density',
             # '25km',
             '100km',
             'testing_learned_rsma_power_common_sweep_1_12.gzip'),
    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6
    # plot_height = plot_width * 0.45

    plot_legend = [
        r'RSMA $\alpha*$',
        r'L-SDMA',
        r'L-RSMA',
        r'H-RSMA',
    ]
    plot_markerstyle = [ 'o', 's', 'd', 'x']
    plot_colors = [ change_lightness(plot_cfg.cp2['black'], 1), plot_cfg.cp3['blue2'], change_lightness(plot_cfg.cp3['red2'], 1), plot_cfg.cp3['red1']]
    plot_linestyles = [ '-', '-', '-', '-',]

    plot_user_number_sweep_testing_graph(
        paths=data_paths,
        metric='sumrate',
        name='user_sweep_test_long',
        width=plot_width,
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    plt.show()