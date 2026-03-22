"""
Comprehensive Evaluation & Outlier Detection for ConvRNNHybrid
=============================================================
Generates:
  1. Per-class Precision / Recall / F1-Score reports
  2. Confusion matrices (saved as PNG images)
  3. Top-k accuracy
  4. Outlier analysis (high-confidence misclassifications with image paths)
  5. JSON + text summary of all results
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Optional: matplotlib for confusion matrix plots
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Warning: matplotlib/seaborn not found. Confusion matrix plots will be skipped.")

from data_loader import get_dataloaders, WikiArtMultiLabelDataset
from model import ConvRNNHybrid

# ===================================================================
# Class Name Mappings (from WikiArt Dataset class files)
# ===================================================================
STYLE_CLASSES = [
    'Abstract Expressionism', 'Action Painting', 'Analytical Cubism',
    'Art Nouveau', 'Baroque', 'Color Field Painting', 'Contemporary Realism',
    'Cubism', 'Early Renaissance', 'Expressionism', 'Fauvism',
    'High Renaissance', 'Impressionism', 'Mannerism (Late Renaissance)',
    'Minimalism', 'Naive Art / Primitivism', 'New Realism',
    'Northern Renaissance', 'Pointillism', 'Pop Art', 'Post-Impressionism',
    'Realism', 'Rococo', 'Romanticism', 'Symbolism', 'Synthetic Cubism',
    'Ukiyo-e'
]

ARTIST_CLASSES = [
    'Albrecht Durer', 'Boris Kustodiev', 'Camille Pissarro', 'Childe Hassam',
    'Claude Monet', 'Edgar Degas', 'Eugene Boudin', 'Gustave Dore',
    'Ilya Repin', 'Ivan Aivazovsky', 'Ivan Shishkin', 'John Singer Sargent',
    'Marc Chagall', 'Martiros Saryan', 'Nicholas Roerich', 'Pablo Picasso',
    'Paul Cezanne', 'Pierre-Auguste Renoir', 'Pyotr Konchalovsky',
    'Raphael Kirchner', 'Rembrandt', 'Salvador Dali', 'Vincent van Gogh'
]

GENRE_CLASSES = [
    'Abstract Painting', 'Cityscape', 'Genre Painting', 'Illustration',
    'Landscape', 'Nude Painting', 'Portrait', 'Religious Painting',
    'Sketch and Study', 'Still Life'
]


def get_class_names(task):
    return {'style': STYLE_CLASSES, 'artist': ARTIST_CLASSES, 'genre': GENRE_CLASSES}[task]


def get_num_classes(task):
    return len(get_class_names(task))


# ===================================================================
# Evaluation
# ===================================================================
def run_evaluation(model, dataloader, device, dataset):
    """Run full evaluation and collect all predictions."""
    model.eval()

    results = {
        'style':  {'preds': [], 'true': [], 'confs': [], 'probs': []},
        'artist': {'preds': [], 'true': [], 'confs': [], 'probs': []},
        'genre':  {'preds': [], 'true': [], 'confs': [], 'probs': []},
    }
    image_paths = []

    sample_idx = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            batch_size = images.size(0)

            with torch.amp.autocast('cuda'):
                outputs = model(images)

            for task in ['style', 'artist', 'genre']:
                probs = F.softmax(outputs[task], dim=1)
                conf, pred = torch.max(probs, dim=1)
                true = labels[task].to(device)

                results[task]['preds'].extend(pred.cpu().numpy())
                results[task]['true'].extend(true.cpu().numpy())
                results[task]['confs'].extend(conf.cpu().numpy())
                results[task]['probs'].extend(probs.cpu().numpy())

            for i in range(batch_size):
                idx = sample_idx + i
                if idx < len(dataset.data):
                    image_paths.append(dataset.data[idx]['image_path'])
                else:
                    image_paths.append(f"unknown_{idx}")
            sample_idx += batch_size

    for task in results:
        results[task]['preds'] = np.array(results[task]['preds'])
        results[task]['true'] = np.array(results[task]['true'])
        results[task]['confs'] = np.array(results[task]['confs'])
        results[task]['probs'] = np.array(results[task]['probs'])

    return results, image_paths


# ===================================================================
# Metrics
# ===================================================================
def compute_top_k_accuracy(true_labels, probs, k):
    """Compute top-k accuracy manually."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]  # indices of top-k predictions
    correct = 0
    for i in range(len(true_labels)):
        if true_labels[i] in top_k_preds[i]:
            correct += 1
    return correct / len(true_labels)


def compute_metrics(results, output_dir):
    """Compute and save per-class metrics + classification reports."""
    os.makedirs(output_dir, exist_ok=True)
    summary = {}

    for task in ['style', 'artist', 'genre']:
        preds = results[task]['preds']
        true = results[task]['true']
        probs = results[task]['probs']
        class_names = get_class_names(task)
        num_classes = len(class_names)
        all_labels = list(range(num_classes))

        # Overall accuracy
        accuracy = np.mean(preds == true)

        # Top-k accuracy
        top3_acc = compute_top_k_accuracy(true, probs, k=min(3, num_classes))
        top5_acc = compute_top_k_accuracy(true, probs, k=min(5, num_classes))

        # Macro averages (use labels param to handle missing classes)
        macro_f1 = f1_score(true, preds, average='macro', labels=all_labels, zero_division=0)
        macro_precision = precision_score(true, preds, average='macro', labels=all_labels, zero_division=0)
        macro_recall = recall_score(true, preds, average='macro', labels=all_labels, zero_division=0)

        # Per-class report — use labels param to match target_names
        report_str = classification_report(
            true, preds,
            labels=all_labels,
            target_names=class_names,
            zero_division=0,
        )

        # Save report to file
        report_path = os.path.join(output_dir, f'{task}_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"{task.upper()} Classification Report\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Top-1 Accuracy:  {accuracy*100:.2f}%\n")
            f.write(f"Top-3 Accuracy:  {top3_acc*100:.2f}%\n")
            f.write(f"Top-5 Accuracy:  {top5_acc*100:.2f}%\n")
            f.write(f"Macro F1:        {macro_f1*100:.2f}%\n")
            f.write(f"Macro Precision: {macro_precision*100:.2f}%\n")
            f.write(f"Macro Recall:    {macro_recall*100:.2f}%\n\n")
            f.write(report_str)

        summary[task] = {
            'accuracy': float(accuracy),
            'top3_accuracy': float(top3_acc),
            'top5_accuracy': float(top5_acc),
            'macro_f1': float(macro_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
        }

        print(f"\n{'='*60}")
        print(f"  {task.upper()} METRICS")
        print(f"{'='*60}")
        print(f"  Top-1 Accuracy : {accuracy*100:.2f}%")
        print(f"  Top-3 Accuracy : {top3_acc*100:.2f}%")
        print(f"  Top-5 Accuracy : {top5_acc*100:.2f}%")
        print(f"  Macro F1       : {macro_f1*100:.2f}%")
        print(f"  Macro Precision: {macro_precision*100:.2f}%")
        print(f"  Macro Recall   : {macro_recall*100:.2f}%")
        print(f"  Report saved   : {report_path}")

    # Save summary JSON
    summary_path = os.path.join(output_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\nMetrics summary saved to {summary_path}")

    return summary


# ===================================================================
# Confusion Matrices
# ===================================================================
def plot_confusion_matrices(results, output_dir):
    """Generate and save confusion matrix heatmaps."""
    if not HAS_PLOT:
        print("Skipping confusion matrix plots (matplotlib not available).")
        return

    os.makedirs(output_dir, exist_ok=True)

    for task in ['style', 'artist', 'genre']:
        preds = results[task]['preds']
        true = results[task]['true']
        class_names = get_class_names(task)
        num_classes = len(class_names)

        cm = confusion_matrix(true, preds, labels=range(num_classes))

        # Normalize
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_normalized = cm.astype('float') / row_sums

        fig_size = max(8, num_classes * 0.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        sns.heatmap(
            cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, vmin=0, vmax=1,
            annot_kws={'size': 7} if num_classes > 15 else {}
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(f'{task.capitalize()} Confusion Matrix (Normalized)', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'{task}_confusion_matrix.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved: {save_path}")


# ===================================================================
# Outlier Detection
# ===================================================================
def find_outliers(results, image_paths, output_dir, confidence_threshold=0.85, max_outliers=50):
    """
    Find paintings misclassified with high confidence.
    These are 'outliers' — paintings that don't fit their assigned category.
    """
    os.makedirs(output_dir, exist_ok=True)
    outliers = []

    num_samples = len(image_paths)

    for i in range(num_samples):
        outlier_info = {'image_path': image_paths[i], 'mismatches': []}

        for task in ['style', 'artist', 'genre']:
            pred = int(results[task]['preds'][i])
            true = int(results[task]['true'][i])
            conf = float(results[task]['confs'][i])
            class_names = get_class_names(task)

            if pred != true and conf > confidence_threshold:
                pred_name = class_names[pred] if pred < len(class_names) else f"class_{pred}"
                true_name = class_names[true] if true < len(class_names) else f"class_{true}"

                outlier_info['mismatches'].append({
                    'task': task,
                    'true_label': true_name,
                    'true_index': true,
                    'predicted_label': pred_name,
                    'predicted_index': pred,
                    'confidence': round(conf, 4),
                })

        if outlier_info['mismatches']:
            outliers.append(outlier_info)

    # Sort by highest confidence
    outliers.sort(key=lambda x: max(m['confidence'] for m in x['mismatches']), reverse=True)
    outliers = outliers[:max_outliers]

    # Save JSON
    outlier_path = os.path.join(output_dir, 'outliers.json')
    with open(outlier_path, 'w') as f:
        json.dump(outliers, f, indent=4)

    # Save readable text report
    report_path = os.path.join(output_dir, 'outlier_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("OUTLIER ANALYSIS REPORT\n")
        f.write("Paintings misclassified with high confidence\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n")
        f.write("=" * 70 + "\n\n")

        for idx, outlier in enumerate(outliers):
            f.write(f"--- Outlier #{idx+1} ---\n")
            f.write(f"Image: {outlier['image_path']}\n")
            for m in outlier['mismatches']:
                f.write(f"  [{m['task'].upper()}] True: {m['true_label']} -> "
                        f"Predicted: {m['predicted_label']} "
                        f"(confidence: {m['confidence']*100:.1f}%)\n")
                f.write(f"    Interpretation: This painting is labeled as "
                        f"'{m['true_label']}' but the model is {m['confidence']*100:.1f}% "
                        f"confident it belongs to '{m['predicted_label']}'. "
                        f"This suggests it may share visual characteristics "
                        f"with {m['predicted_label']} works.\n")
            f.write("\n")

    print(f"\nFound {len(outliers)} high-confidence outliers.")
    print(f"Outlier JSON saved: {outlier_path}")
    print(f"Outlier report saved: {report_path}")

    return outliers


# ===================================================================
# Main
# ===================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory
    if os.path.exists('/content/drive/MyDrive'):
        output_dir = '/content/drive/MyDrive/ArtGAN_Results/evaluation'
    else:
        output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # --- Load Data ---
    print("\nLoading validation data...")
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = WikiArtMultiLabelDataset(
        root_dir='.', split='val', transform=val_transform, mock_images=False
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # --- Load Model ---
    print("Loading trained model...")
    model = ConvRNNHybrid(
        num_styles=27, num_artists=23, num_genres=10,
        hidden_dim=512, num_rnn_layers=2, dropout=0.3, freeze_backbone=True
    )

    ckpt_paths = [
        '/content/drive/MyDrive/ArtGAN_Results/checkpoints/best_model.pth',
        'checkpoints/best_model.pth',
    ]
    loaded = False
    for ckpt_path in ckpt_paths:
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Checkpoint from epoch {checkpoint.get('epoch', '?')}, "
                      f"val_loss={checkpoint.get('val_loss', '?')}")
            else:
                model.load_state_dict(checkpoint)
            loaded = True
            break

    if not loaded:
        print("ERROR: No checkpoint found!")
        return

    model = model.to(device)

    # --- Run Evaluation ---
    print("\n" + "=" * 60)
    print("  RUNNING COMPREHENSIVE EVALUATION")
    print("=" * 60)

    results, image_paths = run_evaluation(model, val_loader, device, val_dataset)

    # --- Compute Metrics ---
    summary = compute_metrics(results, output_dir)

    # --- Confusion Matrices ---
    print("\nGenerating confusion matrices...")
    plot_confusion_matrices(results, output_dir)

    # --- Outlier Detection ---
    print("\nRunning outlier detection...")
    outliers = find_outliers(results, image_paths, output_dir,
                             confidence_threshold=0.85, max_outliers=50)

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    for task in ['style', 'artist', 'genre']:
        s = summary[task]
        print(f"  {task.upper():8s} | Acc: {s['accuracy']*100:.1f}% | "
              f"Top-3: {s['top3_accuracy']*100:.1f}% | "
              f"Top-5: {s['top5_accuracy']*100:.1f}% | "
              f"F1: {s['macro_f1']*100:.1f}%")
    print(f"\n  Outliers found: {len(outliers)}")
    print(f"  All results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
