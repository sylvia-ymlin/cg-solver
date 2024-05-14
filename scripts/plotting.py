import matplotlib.pyplot as plt
import json

PALETTE = {
    "gray_light": "#D8DEE9",
    "gray_dark":  "#4C566A",
    "blue_light": "#88C0D0",
    "blue_mid":   "#81A1C1",
    "blue_deep":  "#5E81AC",
    "orange":     "#D08770",
    "red":        "#BF616A",
}

plt.rcParams.update({
    'font.family': 'monospace',
    'font.size': 11,
    'lines.linewidth': 2,
    'lines.markersize': 7,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': PALETTE['gray_dark'],
    'axes.labelcolor': PALETTE['gray_dark'],
    'xtick.color': PALETTE['gray_dark'],
    'ytick.color': PALETTE['gray_dark'],
    'text.color': PALETTE['gray_dark'],
    'legend.frameon': False,
})


def setup_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=PALETTE['gray_light'], linewidth=0.8)
    ax.spines['left'].set_color(PALETTE['gray_light'])
    ax.spines['bottom'].set_color(PALETTE['gray_light'])


def plot_weak_scaling(data_file='results/weak_scaling.json', output_file='results/weak_scaling.png'):
    with open(data_file) as f:
        data = json.load(f)

    nps = [r['np'] for r in data]
    times = [r['time'] for r in data]
    weak_efficiency = [times[0] / t for t in times]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(nps, weak_efficiency, 'o-', color=PALETTE['blue_mid'], label='Measured')
    ax.plot(nps, [1.0] * len(nps), '--', color=PALETTE['gray_light'], label='Ideal')
    ax.legend()
    setup_ax(ax, 'Weak Scaling (512×512 per rank)', 'Number of Processes', 'Weak Scaling Efficiency (T1/Tp)')
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")


def plot_strong_scaling(data_file='results/strong_scaling.json', output_file='results/strong_scaling.png'):
    with open(data_file) as f:
        data = json.load(f)

    nps = [r['np'] for r in data]
    times = [r['time'] for r in data]
    speedup = [times[0] / t for t in times]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(nps, speedup, 'o-', color=PALETTE['blue_deep'], label='Measured')
    ax.plot(nps, nps, '--', color=PALETTE['gray_light'], label='Ideal')
    ax.legend()
    setup_ax(ax, 'Strong Scaling (2048×2048)', 'Number of Processes', 'Speedup (T1/Tp)')
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")


def plot_convergence(data_file='results/convergence.json', output_file='results/convergence_analysis.png'):
    import numpy as np
    with open(data_file) as f:
        data = json.load(f)

    ns = np.array([r['n'] for r in data])
    times = [r['time'] for r in data]
    iters = [r['iters'] for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(ns, times, 'o-', color=PALETTE['blue_mid'], label='Measured')
    ref_time = ns**3 / ns[0]**3 * times[0]
    ax1.loglog(ns, ref_time, '--', color=PALETTE['gray_light'], label='$O(n^3)$')
    ax1.legend()
    setup_ax(ax1, 'Time Scaling', 'Grid Size N', 'Total Time (s)')

    ax2.plot(ns, iters, 's-', color=PALETTE['orange'], label='Measured')
    ref_iters = ns / ns[0] * iters[0]
    ax2.plot(ns, ref_iters, '--', color=PALETTE['gray_light'], label='$O(n)$')
    ax2.legend()
    setup_ax(ax2, 'Iteration Count', 'Grid Size N', 'Iterations')

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")


def plot_jacobi_tuning(data_file='results/jacobi_tuning.json', output_file='results/jacobi_tuning.png'):
    with open(data_file) as f:
        data = json.load(f)

    steps = [x['jacobi_steps'] for x in data]
    iters = [x['iters'] for x in data]
    times = [x['time'] for x in data]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(steps, iters, 'o-', color=PALETTE['blue_deep'])
    setup_ax(ax1, 'Jacobi Tuning (N=1024)', 'Jacobi Steps', 'Iterations')
    ax1.yaxis.label.set_color(PALETTE['blue_deep'])
    ax1.tick_params(axis='y', labelcolor=PALETTE['blue_deep'])

    ax2 = ax1.twinx()
    ax2.plot(steps, times, 's--', color=PALETTE['red'])
    ax2.set_ylabel('Total Time (s)', color=PALETTE['red'])
    ax2.tick_params(axis='y', labelcolor=PALETTE['red'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color(PALETTE['gray_light'])

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")
