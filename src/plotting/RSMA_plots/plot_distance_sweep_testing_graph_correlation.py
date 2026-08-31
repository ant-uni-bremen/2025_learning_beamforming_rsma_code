# import gzip
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import numpy as np
# from pathlib import Path
#
# from src.config.config import Config
# from src.config.config_plotting import (
#     PlotConfig,
#     save_figures,
#     generic_styling,
#     change_lightness,
# )
#
#
# def plot_distance_sweep_testing_graph(
#         paths,
#         name,
#         width,
#         height,
#         plots_parent_path,
#         legend: list | None = None,
#         colors: list | None = None,
#         markerstyle: list | None = None,
#         linestyles: list | None = None,
#         power_factor_from_path_idx: int = 4,
# ) -> None:
#
#
#
#     y_label = 'Correlation $\\rho_{k,i}$'
#
#     # --- load all data
#     data = []
#     for path in paths:
#         with gzip.open(path, 'rb') as f:
#             data.append(pickle.load(f))   # [x, metrics_dict]
#
#
#
#     fig, ax = plt.subplots(figsize=(width, height))
#
#     # --- main metric curves (bottom axis)
#     step = 20
#     offsets = [0, 5, 10]
#
#     for data_id, (x, metrics_dict) in enumerate(data):
#         # metric_key = get_metric_key(metrics_dict, match_string)
#
#         marker = markerstyle[data_id] if markerstyle is not None else None
#         if marker in ('None', 'none'):
#             marker = None
#
#         color = colors[data_id] if colors is not None else None
#         linestyle = linestyles[data_id] if linestyles is not None else None
#
#         offset = offsets[data_id] % step if data_id < len(offsets) else (data_id * 3) % step
#         markevery = (offset, step)
#
#         hollow = (marker not in (None, '') and marker != 'x')
#         mfc = 'none' if hollow else None
#
#         # Genie betonen
#         is_genie = (data_id == power_factor_from_path_idx)
#         lw = 2.0 if is_genie else 1.4
#
#         ax.plot(
#             x,
#             # metrics_dict[metric_key]['mean'],
#             metrics_dict['correlation']['mean'],
#             color=color,
#             linestyle=linestyle,
#             linewidth=lw,
#             marker=plot_markerstyle,
#             markeredgecolor=marker_edge_colors,
#             markerfacecolor=marker_fill_colors,
#             markeredgewidth=1.5,
#             markersize=7,
#             markevery=markevery,
#         )
#
#     ax.set_ylabel(y_label)
#     ax.set_xlabel('User Distance $d_\mathcal{K}$ [km]')
#     ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x / 1000:g}'))
#     ax.set_ylim(0, 1.1)
#     ax.set_yticks([0, 1 / 2, 1])
#
#     if legend:
#         ax.legend(legend, ncols=1,loc='upper right')
#
#
#     # # --- alpha curve (top axis) from selected dataset (genie)
#
#     generic_styling(ax=ax)
#     fig.tight_layout(pad=0)
#
#
#     save_figures(plots_parent_path=plots_parent_path, plot_name=name + '_' , padding=0.05)
#
#
# if __name__ == '__main__':
#     cfg = Config()
#     plot_cfg = PlotConfig()
#
#     data_paths = [
#         Path(cfg.output_metrics_path, 'channel_correlation_perfect', 'user_distance_sweep',
#              '1sat_8ant_100k_500-50k_0_1.gzip'),
#         Path(cfg.output_metrics_path, 'channel_correlation_perfect', 'user_distance_sweep',
#              '1sat_8ant_100k_500-50k_0_2.gzip'),
#         Path(cfg.output_metrics_path, 'channel_correlation_perfect', 'user_distance_sweep',
#              '1sat_8ant_100k_500-50k_1_2.gzip'),
#         # Path(cfg.output_metrics_path, 'channel_correlation', 'user_distance_sweep',
#         #      '1sat_8ant_100k_500-50k_overall.gzip'),
#
#     ]
#
#     plot_width = 0.99 * plot_cfg.textwidth
#     plot_height = plot_width * 0.45
#
#     plot_legend = [
#         r'$k=1,\  i = 2$ ',
#         r'$k=1,\  i = 3$',
#         r'$k=2,\  i = 3$',
#         # r'overall',
#
#     ]
#
#     plot_markerstyle = ['o', '^', 's']  # besser als 'x', weil 'x' keine Füllung hat
#
#     marker_edge_colors = ['blue', 'blue', 'red']
#     marker_fill_colors = ['red', 'black', 'black']
#
#     plot_colors = [
#         change_lightness(plot_cfg.cp3['black'], 1),
#         change_lightness(plot_cfg.cp3['black'], 1),
#         change_lightness(plot_cfg.cp3['black'], 1),
#         plot_cfg.cp2['black'],
#     ]
#
#     plot_linestyles = ['-', '-', '-']
#
#     plot_distance_sweep_testing_graph(
#         paths=data_paths,
#         name='dist_sweep_correlation',
#         width=plot_width,
#         height=plot_height,
#         legend=plot_legend,
#         colors=plot_colors,
#         markerstyle=plot_markerstyle,
#         markeredgecolor = marker_edge_colors,
#         markerfacecolor = marker_fill_colors,
#         linestyles=plot_linestyles,
#         plots_parent_path=plot_cfg.plots_parent_path,
#         power_factor_from_path_idx=4,
#     )
#
#     plt.show()

import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from src.config.config import Config
from src.config.config_plotting import (
    PlotConfig,
    save_figures,
    generic_styling,
    change_lightness,
)


def plot_distance_sweep_testing_graph(
        paths,
        name,
        width,
        height,
        plots_parent_path,
        legend: list | None = None,
        colors: list | None = None,
        markerstyle: list | None = None,
        marker_edge_colors: list | None = None,
        marker_fill_colors: list | None = None,
        linestyles: list | None = None,
        power_factor_from_path_idx: int = 4,
) -> None:

    y_label = r'User Correlation'

    # --- load all data
    data = []
    for path in paths:
        with gzip.open(path, 'rb') as f:
            data.append(pickle.load(f))   # [x, metrics_dict]

    fig, ax = plt.subplots(figsize=(width, height))

    # --- marker settings
    step = 20
    offsets = [10, 0]

    for data_id, (x, metrics_dict) in enumerate(data):

        # Marker auswählen
        marker = markerstyle[data_id] if markerstyle is not None else None
        if marker in ('None', 'none', ''):
            marker = None

        # Linienfarbe auswählen
        color = colors[data_id] if colors is not None else None

        # Linienstil auswählen
        linestyle = linestyles[data_id] if linestyles is not None else '-'

        # Marker-Randfarbe auswählen
        if marker_edge_colors is not None:
            marker_edge_color = marker_edge_colors[data_id]
        else:
            marker_edge_color = color

        # Marker-Füllfarbe auswählen
        if marker_fill_colors is not None:
            marker_fill_color = marker_fill_colors[data_id]
        else:
            marker_fill_color = color

        # Markerpositionen leicht versetzen,
        # damit überlappende Kurven besser sichtbar sind
        offset = offsets[data_id] % step if data_id < len(offsets) else (data_id * 3) % step
        markevery = (offset, step)

        # Genie betonen, falls vorhanden
        is_genie = (data_id == power_factor_from_path_idx)
        lw = 2.0 if is_genie else 1.4

        label = legend[data_id] if legend is not None else None

        ax.plot(
            x,
            metrics_dict['correlation']['mean'],
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            marker=marker,
            markeredgecolor=marker_edge_color,
            markerfacecolor=marker_fill_color,
            markeredgewidth=1.5,
            markersize=5,
            markevery=markevery,
            label=label,
        )

    ax.set_ylabel(y_label)
    ax.set_xlabel(r'User Distance $d_\mathcal{K}$ [km]')

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f'{x / 1000:g}')
    )

    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 1 / 2, 1])

    if legend:
        ax.legend(ncols=1, loc='upper right')

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    save_figures(
        plots_parent_path=plots_parent_path,
        plot_name=name + '_',
        padding=0.05
    )


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(
            cfg.output_metrics_path,
            'channel_correlation_perfect',
            'user_distance_sweep',
            '1sat_8ant_100k_500-50k_0_1.gzip'
        ),
        Path(
            cfg.output_metrics_path,
            'channel_correlation_perfect',
            'user_distance_sweep',
            '1sat_8ant_100k_500-50k_0_2.gzip'
        ),
        # Path(
        #     cfg.output_metrics_path,
        #     'channel_correlation_perfect',
        #     'user_distance_sweep',
        #     '1sat_8ant_100k_500-50k_1_2.gzip'
        # ),
    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.4

    plot_legend = [
        r'$\rho_{1,2} = \rho_{2,3}$',
        r'$\rho_{1,3}$',
        r'$\rho_{2,3}$',
    ]

    # Marker mit Fläche verwenden.
    # Kein 'x', weil 'x' keine Innenfläche hat.
    plot_markerstyle = ['o', 'd', 's']

    # Gewünschte Marker-Farbkodierung:
    # Marker 1: außen blau, innen rot
    # Marker 2: außen blau, innen schwarz
    # Marker 3: außen rot, innen schwarz
    marker_edge_colors = [
        'black',
        'black',
        'black',
    ]

    marker_fill_colors = [
        'black',
        'black',
        'black',
    ]

    # Linien bleiben schwarz
    plot_colors = [
        change_lightness(plot_cfg.cp3['black'], 1),
        change_lightness(plot_cfg.cp3['black'], 1),
        change_lightness(plot_cfg.cp3['black'], 1),
    ]

    plot_linestyles = ['-', '-', '-']

    plot_distance_sweep_testing_graph(
        paths=data_paths,
        name='dist_sweep_correlation',
        width=plot_width,
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        marker_edge_colors=marker_edge_colors,
        marker_fill_colors=marker_fill_colors,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
        power_factor_from_path_idx=4,
    )

    plt.show()