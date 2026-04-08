import scanpy as sc
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr

def evaluate_representation(adata, z, marker_genes=None):
    """
    Measures the quality of the learned latent space Z.
    """
    # 1. Store the representation in the AnnData object
    adata.obsm['X_latent'] = z.detach().cpu().numpy() if hasattr(z, 'detach') else z
    
    # 2. Compute Neighbors and Clustering in Latent Space
    # We use Leiden clustering to find 'natural' groups without labels
    sc.pp.neighbors(adata, use_rep='X_latent')
    sc.tl.leiden(adata, key_added='latent_clusters', resolution=0.5)
    
    # 3. Metric: Silhouette Score (Intrinsic Quality)
    # Measures how similar a cell is to its own cluster vs. others
    # Range: [-1, 1]. High positive values = well-defined states.
    asw = silhouette_score(adata.obsm['X_latent'], adata.obs['latent_clusters'])
    
    # 4. Metric: Neighborhood Conservation
    # We check if the neighbors in PCA space (rawish) are still neighbors in Latent space
    # High correlation means the model didn't 'break' the biological manifold
    sc.pp.pca(adata)
    orig_dist = np.linalg.norm(adata.obsm['X_pca'][:, :2], axis=1)
    latent_dist = np.linalg.norm(adata.obsm['X_latent'][:, :2], axis=1)
    manifold_corr, _ = spearmanr(orig_dist, latent_dist)
    
    # 5. Metric: Marker Gene Specificity (Biological Check)
    # If we have marker genes, we check if they are "enriched" in the new clusters
    marker_results = {}
    if marker_genes:
        sc.tl.rank_genes_groups(adata, 'latent_clusters', method='wilcoxon')
        # Check if our markers appear in the top 50 genes for any cluster
        for gene in marker_genes:
            found = False
            for cluster in adata.obs['latent_clusters'].unique():
                top_genes = adata.uns['rank_genes_groups']['names'][cluster]
                if gene in top_genes[:50]:
                    found = True
                    break
            marker_results[gene] = "Found" if found else "Missing"

    return {
        "Silhouette Score": asw,
        "Manifold Conservation (Spearman)": manifold_corr,
        "Marker Consistency": marker_results
    }

# Example Usage:
# Define a few standard PBMC markers (T-cells, B-cells, Monocytes)
my_markers = ['CD3E', 'CD79A', 'LYZ', 'GNLY'] 
results = evaluate_representation(rna, latent_z, marker_genes=my_markers)

print(f"Silhouette (Higher is better): {results['Silhouette Score']:.4f}")
print(f"Manifold Corr (Higher is better): {results['Manifold Conservation (Spearman)']:.4f}")
print("Marker Check:", results['Marker Consistency'])