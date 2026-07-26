"""Training / evaluation loops and metrics."""
import math
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)
from spikingjelly.activation_based import functional


def compute_metrics(y_true, y_pred, target_names):
    oa = accuracy_score(y_true, y_pred) * 100
    kappa = cohen_kappa_score(y_true, y_pred) * 100
    labels = list(range(len(target_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    per_class = np.diag(cm) / np.maximum(support, 1) * 100
    aa = per_class[support > 0].mean()
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=target_names, digits=4, zero_division=0)
    return {'OA': oa, 'AA': aa, 'Kappa': kappa, 'Macro_F1': macro_f1,
            'per_class': per_class, 'support': support,
            'confusion_matrix': cm, 'report': report}


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        functional.reset_net(model)

        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        functional.reset_net(model)
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return (running_loss / total, correct / total * 100,
            np.concatenate(all_preds), np.concatenate(all_labels))


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    all_features, all_labels = [], []
    for x, y in loader:
        _, features = model(x.to(device), return_features=True)
        functional.reset_net(model)
        all_features.append(features.float().cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


def train_model(model, train_loader, val_loader, cfg, device, log_every=20):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(cfg.warmup_epochs, 1)
        progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp = cfg.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc, best_state, patience_counter = 0.0, None, 0

    start_time = time.time()
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        for k, v in zip(('train_loss', 'val_loss', 'train_acc', 'val_acc'),
                        (train_loss, val_loss, train_acc, val_acc)):
            history[k].append(v)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{cfg.epochs} | '
                  f'Train: {train_loss:.4f}/{train_acc:.1f}% | '
                  f'Val: {val_loss:.4f}/{val_acc:.1f}% | Best: {best_val_acc:.1f}%',
                  flush=True)

        if patience_counter >= cfg.patience:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    train_time = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    print(f'  Training done in {train_time:.1f}s. Best val acc: {best_val_acc:.2f}%')
    return model, history, train_time
