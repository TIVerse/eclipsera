"""t-SNE for manifold learning."""
from typing import Literal, Optional

import numpy as np

from ..core.base import BaseEstimator
from ..core.utils import check_random_state
from ..core.validation import check_array


class TSNE(BaseEstimator):
    """t-distributed Stochastic Neighbor Embedding.
    
    t-SNE is a tool to visualize high-dimensional data. It converts similarities
    between data points to joint probabilities and tries to minimize the
    Kullback-Leibler divergence between the joint probabilities of the
    low-dimensional embedding and the high-dimensional data.
    
    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity.
    learning_rate : float, default=200.0
        Learning rate for gradient descent.
    n_iter : int, default=1000
        Maximum number of iterations for the optimization.
    metric : str, default='euclidean'
        Distance metric to use.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator.
    verbose : int, default=0
        Verbosity level.
        
    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the embedding vectors.
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.
    n_iter_ : int
        Number of iterations run.
        
    Examples
    --------
    >>> import numpy as np
    >>> from eclipsera.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = TSNE(n_components=2, random_state=0).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    """
    
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        metric: Literal["euclidean"] = "euclidean",
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose
    
    def _compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        sum_X = np.sum(X**2, axis=1)
        D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * X @ X.T
        D = np.maximum(D, 0)  # Numerical stability
        return D
    
    def _compute_joint_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Compute joint probabilities p_ij from distances."""
        n_samples = X.shape[0]
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(X)
        
        # Binary search for beta (precision) values
        target_entropy = np.log(self.perplexity)
        beta = np.ones(n_samples)
        
        # Conditional probabilities
        P = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            # Binary search for beta[i]
            beta_min = -np.inf
            beta_max = np.inf
            
            for _ in range(50):  # Max iterations for binary search
                # Compute probabilities with current beta
                Di = distances[i].copy()
                Di[i] = 0
                
                Pi = np.exp(-Di * beta[i])
                Pi[i] = 0
                sum_Pi = np.sum(Pi)
                
                if sum_Pi == 0:
                    Pi = np.ones(n_samples) / n_samples
                    sum_Pi = 1.0
                
                Pi = Pi / sum_Pi
                
                # Compute entropy
                Pi_nonzero = Pi[Pi > 1e-12]
                entropy = -np.sum(Pi_nonzero * np.log2(Pi_nonzero))
                
                # Check convergence
                entropy_diff = entropy - target_entropy
                if abs(entropy_diff) < 1e-5:
                    break
                
                # Adjust beta
                if entropy_diff > 0:
                    beta_min = beta[i]
                    if beta_max == np.inf:
                        beta[i] *= 2
                    else:
                        beta[i] = (beta[i] + beta_max) / 2
                else:
                    beta_max = beta[i]
                    if beta_min == -np.inf:
                        beta[i] /= 2
                    else:
                        beta[i] = (beta[i] + beta_min) / 2
            
            P[i] = Pi
        
        # Symmetrize and normalize
        P = (P + P.T) / (2 * n_samples)
        P = np.maximum(P, 1e-12)  # Numerical stability
        
        return P
    
    def _compute_low_dim_affinities(self, Y: np.ndarray) -> np.ndarray:
        """Compute low-dimensional affinities (q_ij) using Student t-distribution."""
        n_samples = Y.shape[0]
        
        # Compute pairwise distances in low-dim space
        sum_Y = np.sum(Y**2, axis=1)
        num = 1 / (1 + sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0)
        
        # Normalize
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        
        return Q
    
    def _kl_divergence(self, P: np.ndarray, Q: np.ndarray) -> float:
        """Compute KL divergence between P and Q."""
        return np.sum(P * np.log(P / Q))
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit X into an embedded space and return that transformed output.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        X = check_array(X)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        if n_samples <= self.perplexity:
            raise ValueError(
                f"perplexity must be less than n_samples ({n_samples}), "
                f"got perplexity={self.perplexity}"
            )
        
        random_state = check_random_state(self.random_state)
        
        # Compute high-dimensional affinities
        if self.verbose:
            print("Computing pairwise affinities...")
        
        P = self._compute_joint_probabilities(X)
        
        # Initialize low-dimensional embedding
        Y = random_state.randn(n_samples, self.n_components) * 1e-4
        
        # Gradient descent with momentum
        Y_momentum = np.zeros_like(Y)
        momentum = 0.5
        final_momentum = 0.8
        momentum_switch_iter = 250
        
        for iteration in range(self.n_iter):
            # Compute low-dimensional affinities
            Q = self._compute_low_dim_affinities(Y)
            
            # Compute gradient
            PQ_diff = P - Q
            
            # Compute gradient using affinities
            sum_Y = np.sum(Y**2, axis=1)
            distances = 1 / (1 + sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * Y @ Y.T)
            np.fill_diagonal(distances, 0)
            
            grad = np.zeros_like(Y)
            for i in range(n_samples):
                grad[i] = 4 * np.sum(
                    (PQ_diff[i, :, np.newaxis] * distances[i, :, np.newaxis]) *
                    (Y[i] - Y),
                    axis=0
                )
            
            # Update with momentum
            if iteration < momentum_switch_iter:
                current_momentum = momentum
            else:
                current_momentum = final_momentum
            
            Y_momentum = current_momentum * Y_momentum - self.learning_rate * grad
            Y = Y + Y_momentum
            
            # Center the embedding
            Y = Y - np.mean(Y, axis=0)
            
            if self.verbose and (iteration + 1) % 100 == 0:
                kl_div = self._kl_divergence(P, Q)
                print(f"Iteration {iteration + 1}: KL divergence = {kl_div:.4f}")
        
        # Final KL divergence
        Q = self._compute_low_dim_affinities(Y)
        self.kl_divergence_ = self._kl_divergence(P, Q)
        self.n_iter_ = self.n_iter
        self.embedding_ = Y
        
        return Y
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TSNE":
        """Fit X into an embedded space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : TSNE
            Fitted estimator.
        """
        self.fit_transform(X, y)
        return self
