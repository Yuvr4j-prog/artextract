"""
Similarity Search Engine for Paintings
=======================================
Loads precomputed embeddings and performs fast nearest-neighbor search
using cosine similarity. Generates visual result grids.
"""

import os
import json
import numpy as np
from PIL import Image

# Optional plotting imports
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Warning: matplotlib not found. Visual results will be skipped.")


class SimilarityEngine:
    """
    Fast cosine similarity search engine over painting embeddings.
    """
    
    def __init__(self, embeddings_path):
        """
        Args:
            embeddings_path: Path to the .npz file from feature_extractor.py
        """
        print(f"Loading embeddings from {embeddings_path}...")
        data = np.load(embeddings_path, allow_pickle=True)
        
        self.embeddings = data['embeddings']       # (N, 2048)
        self.image_paths = data['image_paths']     # (N,)
        self.filenames = data['filenames']          # (N,)
        
        # Build filename -> index lookup
        self.name_to_idx = {}
        for i, name in enumerate(self.filenames):
            self.name_to_idx[str(name)] = i
        
        print(f"Loaded {len(self.embeddings)} embeddings of dim {self.embeddings.shape[1]}")
    
    def find_similar(self, query_idx, k=10):
        """
        Find the k most similar paintings to the query.
        
        Args:
            query_idx: Index of the query painting
            k: Number of similar paintings to return
            
        Returns:
            List of dicts with 'index', 'filename', 'path', 'similarity'
        """
        query_embedding = self.embeddings[query_idx]  # (2048,)
        
        # Cosine similarity (embeddings are already L2-normalized)
        similarities = self.embeddings @ query_embedding  # (N,)
        
        # Get top-k+1 (excluding the query itself)
        top_indices = np.argsort(similarities)[::-1][:k + 1]
        
        results = []
        for idx in top_indices:
            if idx == query_idx:
                continue  # Skip self
            results.append({
                'index': int(idx),
                'filename': str(self.filenames[idx]),
                'path': str(self.image_paths[idx]),
                'similarity': float(similarities[idx]),
            })
            if len(results) >= k:
                break
        
        return results
    
    def find_similar_by_embedding(self, query_embedding, k=10):
        """
        Find the k most similar paintings to an arbitrary embedding.
        
        Args:
            query_embedding: (2048,) numpy array, L2-normalized
            k: Number of similar paintings to return
        """
        similarities = self.embeddings @ query_embedding
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'filename': str(self.filenames[idx]),
                'path': str(self.image_paths[idx]),
                'similarity': float(similarities[idx]),
            })
        
        return results
    
    def find_similar_by_name(self, filename, k=10):
        """Find similar paintings given a filename."""
        if filename not in self.name_to_idx:
            raise ValueError(f"Image '{filename}' not found in index. "
                             f"Available: {len(self.name_to_idx)} images.")
        return self.find_similar(self.name_to_idx[filename], k=k)
    
    def batch_search(self, query_indices, k=10):
        """Run similarity search for multiple queries at once."""
        all_results = {}
        for idx in query_indices:
            all_results[int(idx)] = self.find_similar(idx, k=k)
        return all_results


def visualize_results(engine, query_idx, results, output_path, max_display=5):
    """
    Create a visual grid showing query + top similar paintings.
    
    Args:
        engine: SimilarityEngine instance
        query_idx: Index of the query painting
        results: List of result dicts from find_similar
        output_path: Where to save the figure
        max_display: Maximum number of similar paintings to display
    """
    if not HAS_PLOT:
        print("Skipping visualization (matplotlib not available)")
        return
    
    results = results[:max_display]
    n_cols = min(max_display + 1, 6)
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]
    
    # Plot query
    try:
        query_img = Image.open(str(engine.image_paths[query_idx])).convert('RGB')
        axes[0].imshow(query_img)
    except Exception:
        axes[0].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', fontsize=10)
    axes[0].set_title(f"QUERY\n{engine.filenames[query_idx]}", fontsize=8, fontweight='bold')
    axes[0].axis('off')
    
    # Plot similar paintings
    for i, result in enumerate(results):
        ax = axes[i + 1]
        try:
            img = Image.open(result['path']).convert('RGB')
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', fontsize=10)
        
        sim_pct = result['similarity'] * 100
        ax.set_title(f"#{i+1} (sim: {sim_pct:.1f}%)\n{result['filename']}", fontsize=7)
        ax.axis('off')
    
    plt.suptitle('Painting Similarity Search Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved: {output_path}")


def run_demo(embeddings_path, output_dir, num_queries=10, k=10):
    """
    Run a demo: pick random queries and show their most similar paintings.
    
    Args:
        embeddings_path: Path to embeddings .npz file
        output_dir: Directory to save results
        num_queries: Number of random queries to demo
        k: Number of similar paintings per query
    """
    engine = SimilarityEngine(embeddings_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Pick random queries
    np.random.seed(42)
    n = len(engine.embeddings)
    query_indices = np.random.choice(n, size=min(num_queries, n), replace=False)
    
    all_demo_results = []
    
    for i, query_idx in enumerate(query_indices):
        results = engine.find_similar(int(query_idx), k=k)
        
        # Save visualization
        vis_path = os.path.join(output_dir, f'similar_{i+1}.png')
        visualize_results(engine, int(query_idx), results, vis_path, max_display=5)
        
        all_demo_results.append({
            'query_index': int(query_idx),
            'query_filename': str(engine.filenames[query_idx]),
            'similar_paintings': results,
        })
    
    # Save JSON results
    json_path = os.path.join(output_dir, 'similarity_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_demo_results, f, indent=2)
    print(f"\nAll results saved to {json_path}")
    
    return all_demo_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Painting similarity search')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Path to embeddings .npz file')
    parser.add_argument('--output_dir', type=str, default='similarity_results',
                        help='Output directory for results')
    parser.add_argument('--num_queries', type=int, default=10,
                        help='Number of random queries to demo')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of similar paintings per query')
    
    args = parser.parse_args()
    run_demo(
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        num_queries=args.num_queries,
        k=args.k,
    )
