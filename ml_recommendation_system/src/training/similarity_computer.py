"""
Similarity Computer Module
Computes item-item and user-user similarity matrices
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityComputer:
    """Compute similarity matrices for collaborative filtering"""
    
    def __init__(self, 
                 similarity_metric: str = 'cosine',
                 top_n_items: int = 50,
                 top_n_users: int = 100):
        """
        Initialize SimilarityComputer
        
        Args:
            similarity_metric: Similarity metric ('cosine', 'pearson', etc.)
            top_n_items: Number of similar items to keep per item
            top_n_users: Number of similar users to keep per user
        """
        self.similarity_metric = similarity_metric
        self.top_n_items = top_n_items
        self.top_n_users = top_n_users
        self.item_similarity = None
        self.user_similarity = None
    
    def compute_item_similarity(self, interaction_matrix: csr_matrix) -> csr_matrix:
        """
        Compute item-item similarity matrix
        
        Args:
            interaction_matrix: User-item interaction matrix (users × items)
            
        Returns:
            Item similarity matrix (items × items)
        """
        logger.info("\n" + "="*60)
        logger.info("COMPUTING ITEM-ITEM SIMILARITY")
        logger.info("="*60)
        
        n_items = interaction_matrix.shape[1]
        logger.info(f"Computing similarity for {n_items:,} products...")
        logger.info(f"Metric: {self.similarity_metric}")
        
        # Compute similarity (transpose to get item-item)
        logger.info("Calculating cosine similarity...")
        self.item_similarity = cosine_similarity(interaction_matrix.T, dense_output=False)
        
        # Keep only top-N similar items per item
        logger.info(f"Keeping top-{self.top_n_items} similar items per item...")
        self.item_similarity = self._keep_top_n(self.item_similarity, self.top_n_items)
        
        # Calculate statistics
        n_nonzero = self.item_similarity.nnz
        total_possible = n_items * n_items
        sparsity = 100 * (1 - n_nonzero / total_possible)
        
        logger.info(f"  Matrix shape: {n_items:,} × {n_items:,}")
        logger.info(f"  Non-zero similarities: {n_nonzero:,}")
        logger.info(f"  Sparsity: {sparsity:.2f}%")
        logger.info(f"  Memory (approx): {self.item_similarity.data.nbytes / 1024 / 1024:.1f} MB")
        logger.info("="*60)
        
        return self.item_similarity
    
    def compute_user_similarity(self, interaction_matrix: csr_matrix) -> csr_matrix:
        """
        Compute user-user similarity matrix
        
        Args:
            interaction_matrix: User-item interaction matrix (users × items)
            
        Returns:
            User similarity matrix (users × users)
        """
        logger.info("\n" + "="*60)
        logger.info("COMPUTING USER-USER SIMILARITY")
        logger.info("="*60)
        
        n_users = interaction_matrix.shape[0]
        logger.info(f"Computing similarity for {n_users:,} customers...")
        logger.info(f"Metric: {self.similarity_metric}")
        logger.info("⚠️  This may take 10-20 minutes for large datasets...")
        
        # Compute similarity
        logger.info("Calculating cosine similarity...")
        self.user_similarity = cosine_similarity(interaction_matrix, dense_output=False)
        
        # Keep only top-N similar users per user
        logger.info(f"Keeping top-{self.top_n_users} similar users per user...")
        self.user_similarity = self._keep_top_n(self.user_similarity, self.top_n_users)
        
        # Calculate statistics
        n_nonzero = self.user_similarity.nnz
        total_possible = n_users * n_users
        sparsity = 100 * (1 - n_nonzero / total_possible)
        
        logger.info(f"  Matrix shape: {n_users:,} × {n_users:,}")
        logger.info(f"  Non-zero similarities: {n_nonzero:,}")
        logger.info(f"  Sparsity: {sparsity:.2f}%")
        logger.info(f"  Memory (approx): {self.user_similarity.data.nbytes / 1024 / 1024:.1f} MB")
        logger.info("="*60)
        
        return self.user_similarity
    
    def _keep_top_n(self, similarity_matrix: csr_matrix, top_n: int) -> csr_matrix:
        """
        Keep only top-N similar items/users per row
        
        Args:
            similarity_matrix: Similarity matrix
            top_n: Number of top similarities to keep
            
        Returns:
            Sparse matrix with only top-N per row
        """
        # Convert to LIL format for efficient row operations
        sim_lil = similarity_matrix.tolil()
        
        for i in range(sim_lil.shape[0]):
            # Get row data
            row_data = sim_lil.data[i]
            row_indices = sim_lil.rows[i]
            
            if len(row_data) > top_n + 1:  # +1 for self-similarity
                # Get top-N indices (excluding self)
                top_indices = np.argsort(row_data)[::-1][:top_n + 1]
                
                # Keep only top-N
                sim_lil.data[i] = [row_data[j] for j in top_indices]
                sim_lil.rows[i] = [row_indices[j] for j in top_indices]
        
        # Convert back to CSR format
        return sim_lil.tocsr()
    
    def compute_all(self, interaction_matrix: csr_matrix) -> Tuple[csr_matrix, csr_matrix]:
        """
        Compute both item and user similarity matrices
        
        Args:
            interaction_matrix: User-item interaction matrix
            
        Returns:
            Tuple of (item_similarity, user_similarity)
        """
        item_sim = self.compute_item_similarity(interaction_matrix)
        user_sim = self.compute_user_similarity(interaction_matrix)
        
        return item_sim, user_sim
