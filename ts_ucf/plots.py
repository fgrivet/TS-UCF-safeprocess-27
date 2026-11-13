import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from . import core


def plot_signals(signal_train, signal_test, figsize=(14, 2), cmap="tab10", point_labels=None):
    k = signal_train.shape[1]

    colors = mpl.colormaps[cmap].colors
    colors_cycle = itertools.cycle(colors)

    x_max = max(len(signal_train), len(signal_test))
    y_min = np.minimum(signal_train.min(axis=0), signal_test.min(axis=0))
    y_max = np.maximum(signal_train.max(axis=0), signal_test.max(axis=0))
    margin = 0.2
    y_min_adjusted = y_min - margin * y_min * np.sign(y_min)
    y_max_adjusted = y_max + margin * y_max * np.sign(y_max)

    if point_labels is not None:
        ano_intervals = find_continous_intervals(np.flatnonzero(point_labels))
    else:
        ano_intervals = []

    for num_channel in range(k):
        color = next(colors_cycle)

        plt.figure(figsize=figsize)
        plt.plot(signal_train[:, num_channel], color=color, lw=4, label="Train", alpha=0.4)
        plt.plot(signal_test[:, num_channel], color=color, label="Test")

        for start, end in ano_intervals:
            plt.axvspan(start, end, color="red", alpha=0.5)

        ax = plt.gca()
        ax.set_facecolor("#f0f0f0")
        ax.spines[:].set_visible(False)
        ax.spines["left"].set(visible=True, linewidth=2, bounds=(y_min_adjusted[num_channel], y_max_adjusted[num_channel]))

        ax.yaxis.set_label_text(f"Channel {num_channel+1}")
        ax.yaxis.label.set(fontweight="bold", rotation=0)
        ax.yaxis.set_label_coords(-0.045, 0.5)

        ax.set_yticks([])
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min_adjusted[num_channel], y_max_adjusted[num_channel])

        plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout()
        plt.show()


def plot_windows(windows_train, windows_test, figsize=(14, 2), cmap="tab10"):
    _, L, k = windows_train.shape
    K_test = len(windows_test)

    colors = mpl.colormaps[cmap].colors

    y_min = min(windows_train.min(), windows_test.min())
    y_min = 0.8 * y_min if y_min >= 0 else 1.2 * y_min
    y_max = max(windows_train.max(), windows_test.max())
    y_max = 0.8 * y_max if y_max < 0 else 1.2 * y_max

    for num_window in range(K_test):
        plt.figure(figsize=figsize)

        colors_cycle = itertools.cycle(colors)
        for num_channel in range(k):
            color = next(colors_cycle)
            for w in windows_train:
                plt.plot(w[:, num_channel], color=color, alpha=0.01, lw=6, zorder=1)

            plt.plot(windows_test[num_window, :, num_channel], color=color, lw=3, zorder=10, label=f"Channel {num_channel+1}")

        ax = plt.gca()
        ax.set_facecolor("#f0f0f0")
        ax.spines[:].set_visible(False)
        ax.spines["left"].set(visible=True, linewidth=2, bounds=(y_min, y_max))

        ax.yaxis.set_label_text(f"Window {num_window+1}")
        ax.yaxis.label.set(fontweight="bold", rotation=0)
        ax.yaxis.set_label_coords(-0.045, 0.5)

        ax.set_yticks([])
        ax.set_xlim(0, L)
        ax.set_ylim(y_min, y_max)

        plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout()
        plt.show()


def plot_distances(distances, figsize=(14, 2), n_poly_opt: int = 0):
    K_test, K_train = distances.shape

    d_min_log = np.log1p(distances.min())
    d_max_log = np.log1p(distances.max())
    num_ticks = int(figsize[0] * 0.5)

    x_ticks_transformed = np.concat([[0], np.linspace(d_min_log, d_max_log, num_ticks)])
    x_ticks = np.expm1(x_ticks_transformed)

    if n_poly_opt > 0:
        poly_opt, correct_scores, inversion_scores = core.compute_optimal_polynomial(distances, n=n_poly_opt)

    for num_window in range(K_test):
        plt.figure(figsize=figsize)

        x = np.log1p(distances[num_window])

        if n_poly_opt > 0:
            y = np.log1p(np.polyval(poly_opt[num_window, ::-1], distances[num_window]) ** 2)
            x_poly = np.linspace(0, 1.2 * distances.max(), 1000)
            y_poly = np.polyval(poly_opt[num_window, ::-1], x_poly) ** 2

            plt.plot(np.log1p(x_poly), np.log1p(y_poly), color="tab:orange", lw=2, label="$Q^{\\star 2}$")

            plt.vlines(x, 0, y, lw=1)

            plt.title(f"score = {correct_scores[num_window, 0]:_.6g} - poly score = {inversion_scores[num_window, 0]:_.6g}")
            print(y.sum(), 1 / y.sum())
            a, b = distances[num_window].min(), distances[num_window].max()
            r = (b - a) / (b + a)
            q = (np.sqrt(b) - np.sqrt(a)) / (np.sqrt(b) + np.sqrt(a))
            q_res = (4 * q**4) / ((1 + q**4) ** 2)
            print(a, b, r, r**4, 1 / (r**4), q_res, 1 / q_res)

        else:
            y = np.ones(K_train)

        plt.scatter(x, y, s=40, label="distances")

        ax = plt.gca()
        ax.set_facecolor("#f0f0f0")
        ax.spines[:].set_visible(False)

        ax.yaxis.set_label_text(f"Window {num_window+1}")
        ax.yaxis.label.set(fontweight="bold", rotation=0)
        ax.yaxis.set_label_coords(-0.07, 0.5)

        if n_poly_opt > 0:
            ax.axhline(y=0, ls="dotted", color="black")
            ax.axhline(y=np.log1p(1), ls="dotted", color="gray")

            y_ticks = np.array([0, 1, y_poly.mean(), y_poly.max()])
            y_ticks_transformed = np.log1p(y_ticks)
            ax.set_yticks(y_ticks_transformed, [f"{t:.4g}" for t in y_ticks])
        else:
            ax.set_yticks([])
            ax.set_ylim(0, 2)
            ax.spines["left"].set(visible=True, linewidth=2, bounds=(0, 2))

        ax.set_xticks(x_ticks_transformed, [f"{t:.4g}" for t in x_ticks], rotation=45)
        ax.set_xlim(0, 1.2 * d_max_log)

        plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout()
        plt.show()


def plot_spectrum_distances(windows_train, windows_test, figsize=(14, 2), cmap="inferno"):
    w_test_fft = np.abs(np.fft.rfft(windows_test, axis=1))
    w_train_fft = np.abs(np.fft.rfft(windows_train, axis=1))

    dist = (w_test_fft[:, None, :, :] - w_train_fft[None, :, :, :]) ** 2  # Shape K_test, K_train, L//2, k
    dist = np.log1p(dist)
    max_dist = dist.max(axis=(0, 1, 2))

    norms = [mpl.colors.Normalize(vmin=0, vmax=max_) for max_ in max_dist]

    K_test, _, k = windows_test.shape
    for num_window in range(K_test):
        plt.figure(figsize=figsize)
        for num_channel in range(k):
            plt.subplot(1, k, num_channel + 1)
            plt.imshow(dist[num_window, :, :, num_channel], cmap=cmap, norm=norms[num_channel])
            plt.colorbar()
            if num_channel == 0:
                ax = plt.gca()
                ax.yaxis.set_label_text(f"Window {num_window+1}")
                ax.yaxis.label.set(fontweight="bold", rotation=0)
                ax.yaxis.set_label_coords(-0.6, 0.5)
        plt.tight_layout()
        plt.show()


def find_continous_intervals(arr):
    if len(arr) == 0:
        return []

    diffs = np.diff(arr)

    break_points = np.where(diffs != 1)[0] + 1

    break_points = np.insert(break_points, 0, 0)
    break_points = np.append(break_points, len(arr))

    # Créer les intervalles
    intervals = []
    for i in range(len(break_points) - 1):
        start = arr[break_points[i]]
        end = arr[break_points[i + 1] - 1]
        intervals.append((start, end))

    return intervals


def find_ano_events(arr):
    diff = np.diff(np.concatenate(([0], arr)))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(ends) < len(starts):
        ends = np.append(ends, len(arr))

    return list(zip(starts, ends))


def plot_scores(signal_test, scores, s, point_labels=None, figsize=(14, 2)):
    scores_r = np.repeat(scores, s, axis=0)

    for num_channel in range(signal_test.shape[1]):
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.plot(signal_test[:, num_channel], color="tab:blue", label="Signal")
        ax.set_ylabel(f"Channel {num_channel+1}", color="tab:blue")
        ax2 = ax.twinx()
        if scores.ndim == 2:
            curr_scores_r = scores_r[:, num_channel]
        else:
            curr_scores_r = scores_r
        ax2.plot(curr_scores_r, color="black", label="Score")
        if point_labels is not None:
            for start, end in find_continous_intervals(np.flatnonzero(point_labels)):
                plt.axvspan(start, end, color="red", alpha=0.5)
        ax2.set_ylabel("Score", color="black")
        ax2.set_yscale("log")
        plt.tight_layout()
        plt.show()
