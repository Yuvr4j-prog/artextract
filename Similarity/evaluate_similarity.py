"""
Evaluation Metrics for Painting Similarity
==========================================
Computes Precision@k, Recall@k, and other retrieval metrics.
Uses metadata (artist, style, medium, etc.) as ground-truth relevance.
"""

import os
import json
import numpy as np
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def build_relevance_groups(metadata_list, group_key='attribution'):
    """
    Build ground-truth relevance groups from metadata.
    Two paintings are "relevant" if they share the same group_key value.
    
    Args:
        metadata_list: List of metadata dicts (one per painting)
        group_key: Which metadata field to use for relevance
                   Options: 'attribution' (artist), 'classification' (medium),
                            'style', 'school', etc.
    
    Returns:
        groups: dict mapping group_name -> set of indices
        labels: array of group labels for each image
    """
    groups = defaultdict(set)
    labels = []
    
    for idx, meta in enumerate(metadata_list):
        label = str(meta.get(group_key, 'unknown')).strip()
        if not label or label == 'nan':
            label = 'unknown'
        groups[label].add(idx)
        labels.append(label)
    
    # Remove 'unknown' group for evaluation (can't judge relevance)
    if 'unknown' in groups:
        del groups['unknown']
    
    print(f"Built {len(groups)} relevance groups using '{group_key}'")
    for name, indices in sorted(groups.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {name}: {len(indices)} paintings")
    if len(groups) > 10:
        print(f"  ... and {len(groups) - 10} more groups")
    
    return groups, labels


def precision_at_k(relevant_set, retrieved_list, k):
    """
    Precision@k: What fraction of the top-k results are relevant?
    
    Args:
        relevant_set: Set of indices that are relevant to the query
        retrieved_list: Ordered list of retrieved indices
        k: Number of results to consider
    """
    if k == 0:
        return 0.0
    top_k = retrieved_list[:k]
    num_relevant = sum(1 for idx in top_k if idx in relevant_set)
    return num_relevant / k


def recall_at_k(relevant_set, retrieved_list, k):
    """
    Recall@k: What fraction of all relevant items appear in the top-k?
    
    Args:
        relevant_set: Set of indices that are relevant to the query
        retrieved_list: Ordered list of retrieved indices
        k: Number of results to consider
    """
    if len(relevant_set) == 0:
        return 0.0
    top_k = retrieved_list[:k]
    num_relevant = sum(1 for idx in top_k if idx in relevant_set)
    return num_relevant / len(relevant_set)


def mean_average_precision(relevant_set, retrieved_list, max_k=None):
    """
    Average Precision for a single query.
    
    Args:
        relevant_set: Set of relevant indices
        retrieved_list: Ordered list of retrieved indices
        max_k: Maximum depth to consider (None = use all)
    """
    if len(relevant_set) == 0:
        return 0.0
    
    if max_k is not None:
        retrieved_list = retrieved_list[:max_k]
    
    precisions = []
    num_relevant = 0
    
    for i, idx in enumerate(retrieved_list):
        if idx in relevant_set:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / len(relevant_set)


def evaluate_similarity(engine, labels, k_values=[1, 3, 5, 10, 20],
                        num_queries=None, output_dir=None):
    """
    Comprehensive evaluation of the similarity search engine.
    
    Args:
        engine: SimilarityEngine instance
        labels: List of group labels (one per image)
        k_values: List of k values for Precision/Recall@k
        num_queries: Number of queries to evaluate (None = all)
        output_dir: Directory to save results
    
    Returns:
        dict with all metrics
    """
    n = len(engine.embeddings)
    labels = np.array(labels)
    max_k = max(k_values)
    
    # Determine valid queries (those with a known, non-unique group)
    valid_indices = []
    for i in range(n):
        label = labels[i]
        if label != 'unknown':
            group_size = np.sum(labels == label)
            if group_size > 1:  # Need at least 1 other relevant item
                valid_indices.append(i)
    
    if num_queries is not None:
        np.random.seed(42)
        valid_indices = np.random.choice(
            valid_indices, size=min(num_queries, len(valid_indices)), replace=False
        ).tolist()
    
    print(f"\nEvaluating on {len(valid_indices)} queries...")
    
    # Collect metrics
    prec_at_k = {k: [] for k in k_values}
    rec_at_k = {k: [] for k in k_values}
    ap_scores = []
    
    for query_idx in valid_indices:
        query_label = labels[query_idx]
        
        # Get relevant set (same label, excluding self)
        relevant = set(np.where(labels == query_label)[0].tolist())
        relevant.discard(query_idx)
        
        if len(relevant) == 0:
            continue
        
        # Get retrieval results
        results = engine.find_similar(query_idx, k=max_k)
        retrieved = [r['index'] for r in results]
        
        # Compute metrics
        for k in k_values:
            prec_at_k[k].append(precision_at_k(relevant, retrieved, k))
            rec_at_k[k].append(recall_at_k(relevant, retrieved, k))
        
        ap_scores.append(mean_average_precision(relevant, retrieved, max_k=max_k))
    
    # Average
    metrics = {
        'num_queries': len(valid_indices),
        'mAP': float(np.mean(ap_scores)) if ap_scores else 0.0,
    }
    
    for k in k_values:
        metrics[f'precision@{k}'] = float(np.mean(prec_at_k[k])) if prec_at_k[k] else 0.0
        metrics[f'recall@{k}'] = float(np.mean(rec_at_k[k])) if rec_at_k[k] else 0.0
    
    # Print report
    print(f"\n{'='*60}")
    print(f"  SIMILARITY EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Queries evaluated: {metrics['num_queries']}")
    print(f"  Mean Average Precision (mAP): {metrics['mAP']*100:.2f}%")
    print()
    
    print(f"  {'k':>4s} | {'Precision@k':>12s} | {'Recall@k':>10s}")
    print(f"  {'-'*4:>4s}-+-{'-'*12:>12s}-+-{'-'*10:>10s}")
    for k in k_values:
        p = metrics[f'precision@{k}'] * 100
        r = metrics[f'recall@{k}'] * 100
        print(f"  {k:4d} | {p:11.2f}% | {r:9.2f}%")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON
        json_path = os.path.join(output_dir, 'similarity_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {json_path}")
        
        # Text report
        report_path = os.path.join(output_dir, 'similarity_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("PAINTING SIMILARITY EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Queries evaluated: {metrics['num_queries']}\n")
            f.write(f"Mean Average Precision (mAP): {metrics['mAP']*100:.2f}%\n\n")
            f.write(f"{'k':>4s} | {'Precision@k':>12s} | {'Recall@k':>10s}\n")
            f.write(f"{'-'*4:>4s}-+-{'-'*12:>12s}-+-{'-'*10:>10s}\n")
            for k in k_values:
                p = metrics[f'precision@{k}'] * 100
                r = metrics[f'recall@{k}'] * 100
                f.write(f"{k:4d} | {p:11.2f}% | {r:9.2f}%\n")
        print(f"Report saved to {report_path}")
        
        # Plot
        if HAS_PLOT:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            precisions = [metrics[f'precision@{k}'] * 100 for k in k_values]
            recalls = [metrics[f'recall@{k}'] * 100 for k in k_values]
            
            ax1.bar(range(len(k_values)), precisions, color='steelblue', alpha=0.8)
            ax1.set_xticks(range(len(k_values)))
            ax1.set_xticklabels([f'P@{k}' for k in k_values])
            ax1.set_ylabel('Precision (%)')
            ax1.set_title('Precision@k')
            ax1.set_ylim(0, 100)
            for i, v in enumerate(precisions):
                ax1.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9)
            
            ax2.bar(range(len(k_values)), recalls, color='coral', alpha=0.8)
            ax2.set_xticks(range(len(k_values)))
            ax2.set_xticklabels([f'R@{k}' for k in k_values])
            ax2.set_ylabel('Recall (%)')
            ax2.set_title('Recall@k')
            ax2.set_ylim(0, 100)
            for i, v in enumerate(recalls):
                ax2.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9)
            
            plt.suptitle(f'Similarity Retrieval Metrics (mAP: {metrics["mAP"]*100:.1f}%)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, 'metrics_plot.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Metrics plot saved to {plot_path}")
    
    return metrics


if __name__ == '__main__':
    print("This module provides evaluation functions for similarity search.")
    print("Use it from the Colab notebook or import evaluate_similarity().")
