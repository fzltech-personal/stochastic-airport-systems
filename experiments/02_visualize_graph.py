"""
Experiment: Visualize the State-Transition Graph and Proto-Value Functions.

This script runs a short simulation to explore the state space, builds the
transition graph, and visualizes both the graph structure and the 
learned spectral embedding (PVFs).
Robustly handles pathing, large graphs, and edge cases.
"""
import sys
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Add project root to path BEFORE importing from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.paths import ProjectPaths


def main():
    data_dir = ProjectPaths.get_data_dir() / "processed"

    print("Loading pre-computed artifacts...")
    basis_functions = np.load(data_dir / "basis_functions.npy")
    with open(data_dir / "state_graph.pkl", "rb") as f:
        G_sub = pickle.load(f)

    sub_n = G_sub.number_of_nodes()

    # 7. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Plot 1: State-Transition Graph ---
    ax1 = axes[0]
    ax1.set_title(f"State-Transition Graph ({sub_n} nodes)")
    
    # Compute degrees for sizing/coloring
    degrees = dict(G_sub.degree(weight='weight'))
    node_colors = list(degrees.values())
    
    # Layout Check: Avoid hanging on large graphs
    LAYOUT_THRESHOLD = 500
    
    if sub_n < LAYOUT_THRESHOLD:
        print(f"Generating spring layout for {sub_n} nodes...")
        pos = nx.spring_layout(G_sub, seed=42, k=0.15)
        
        node_sizes = [v * 5 for v in degrees.values()]
        
        nx.draw_networkx_nodes(
            G_sub, pos, 
            node_size=node_sizes, 
            node_color=node_colors, 
            cmap=plt.cm.viridis, 
            alpha=0.8,
            ax=ax1
        )
        nx.draw_networkx_edges(G_sub, pos, alpha=0.1, ax=ax1)
        ax1.axis('off')
    else:
        print(f"Skipping graph layout (N={sub_n} > {LAYOUT_THRESHOLD}).")
        ax1.text(0.5, 0.5, 
                 f"Graph too large for layout ({sub_n} nodes)\nSee spectral embedding ->", 
                 ha='center', va='center', fontsize=12)
        ax1.axis('off')

    # --- Plot 2: Spectral Embedding (Eigenvector 1 vs 2) ---
    ax2 = axes[1]
    ax2.set_title("Spectral Embedding (PVF 1 vs PVF 2)\nClusters reveal operational bottlenecks")
    
    if basis_functions.shape[1] >= 2:
        x = basis_functions[:, 0]
        y = basis_functions[:, 1]
        
        scatter = ax2.scatter(
            x, y, 
            c=node_colors, 
            cmap=plt.cm.viridis, 
            alpha=0.7,
            s=50
        )
        plt.colorbar(scatter, ax=ax2, label="Visit Frequency (Degree)")
        
        # Use raw strings to fix invalid escape sequence warnings
        ax2.set_xlabel(r"Basis Function $\phi_1$ (Fiedler Vector)")
        ax2.set_ylabel(r"Basis Function $\phi_2$")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Not enough eigenvectors found for 2D plot", ha='center')

    plt.tight_layout()
    
    # Save output
    output_dir = ProjectPaths.get_root() / "experiments/results/plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "graph_visualization.png"
    
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")
    
    # Clean up memory
    plt.close(fig)


if __name__ == "__main__":
    main()
