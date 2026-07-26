"""Plotting utilities (ported from main.py, headless-safe)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader

from .data import AllPixelDataset


def plot_confusion_matrix(cm, target_names, save_path):
    fig, ax = plt.subplots(figsize=(max(10, len(target_names)), max(8, len(target_names) * 0.8)))
    cm_perc = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Spectral',
                xticklabels=target_names, yticklabels=target_names, ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (%)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_tsne(features, labels, target_names, save_path, max_points=3000):
    if len(features) > max_points:
        idx = np.random.default_rng(0).choice(len(features), max_points, replace=False)
        features, labels = features[idx], labels[idx]
    embedded = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(features)
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    cmap = plt.colormaps.get_cmap('nipy_spectral').resampled(len(unique_labels))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = target_names[label] if label < len(target_names) else str(label)
        ax.scatter(embedded[mask, 0], embedded[mask, 1], c=[cmap(i)], s=5, alpha=0.6, label=name)
    ax.legend(markerscale=3, fontsize=8, loc='best')
    ax.set_title('t-SNE Feature Visualization')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in ((ax1, 'loss', 'Loss'), (ax2, 'acc', 'Accuracy (%)')):
        ax.plot(history[f'train_{key}'], label='Train')
        ax.plot(history[f'val_{key}'], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def plot_segmentation_map(model, meta, window_size, device, save_path, batch_size=512):
    gt = meta['gt']
    h, w = gt.shape
    ds = AllPixelDataset(meta['cube'], h, w, window_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    preds = []
    for batch in loader:
        out = model(batch.to(device))
        functional.reset_net(model)
        preds.append(out.argmax(dim=1).cpu().numpy())
    seg_map = np.concatenate(preds).reshape(h, w) + 1
    seg_map[gt == 0] = 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, img, title in ((axes[0], gt, 'Ground Truth'), (axes[1], seg_map, 'Predicted')):
        ax.imshow(img, cmap='nipy_spectral')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return seg_map
