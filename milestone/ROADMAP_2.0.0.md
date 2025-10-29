# Eclipsera 2.0.0 Roadmap - Performance & Scalability Enhancements

## Overview

This roadmap outlines the planned performance and scalability enhancements for Eclipsera 2.0.0, building on the successful FastAlloc integration in v1.x. The focus is on three key areas:

1. **Advanced Object Pooling** - Workspace pooling for iterative algorithms
2. **Parallel Execution** - Thread-safe pooling for multi-core utilization
3. **Memory Optimization** - Intelligent array reuse and buffer management

---

## Phase 1: Workspace Pooling for Iterative Algorithms ✅

**Status**: Completed in v1.2.0  
**Impact**: 5-15% speedup in CV/hyperparameter search

### Implemented Features
- ✅ Object pooling in GridSearchCV/RandomizedSearchCV
- ✅ Object pooling in cross_val_score/cross_validate
- ✅ BaseEstimator.reset() method
- ✅ Graceful fallback when fastalloc unavailable

---

## Phase 2: Iterative Algorithm Workspace Pooling

**Target**: v2.0.0  
**Priority**: High  
**Estimated Impact**: 10-30% speedup for iterative algorithms

### 2.1 t-SNE Workspace Optimization

#### Current State
```python
# In each iteration (~1000 iterations):
for iteration in range(self.n_iter):
    Q = self._compute_low_dim_affinities(Y)  # Allocates NxN matrix
    grad = np.zeros_like(Y)                   # Allocates gradient array
    Y_momentum = np.zeros_like(Y)             # Allocates momentum array
    # ... computation ...
```

**Problem**: Allocates 3 large arrays per iteration = 3000+ allocations per run

#### Proposed Solution
```python
class TSNEWorkspace:
    """Pre-allocated workspace for t-SNE computation."""
    
    def __init__(self, n_samples: int, n_components: int):
        self.n_samples = n_samples
        self.n_components = n_components
        
        # Pre-allocate reusable buffers
        self.Q = np.zeros((n_samples, n_samples))
        self.grad = np.zeros((n_samples, n_components))
        self.Y_momentum = np.zeros((n_samples, n_components))
        self.distances = np.zeros((n_samples, n_samples))
        self.PQ_diff = np.zeros((n_samples, n_samples))
    
    def reset(self):
        """Reset workspace for reuse."""
        self.Q.fill(0)
        self.grad.fill(0)
        self.Y_momentum.fill(0)
        self.distances.fill(0)
        self.PQ_diff.fill(0)
        return self

# Usage in TSNE.fit_transform():
workspace_pool = Pool(
    obj_type=lambda: TSNEWorkspace(n_samples, self.n_components),
    capacity=1,
    reset_method="reset"
)

with workspace_pool.allocate() as ws:
    for iteration in range(self.n_iter):
        # Reuse ws.Q, ws.grad, ws.Y_momentum
        # No allocations in hot loop!
        self._compute_affinities_inplace(Y, ws.Q)
        self._compute_gradient_inplace(Y, ws)
```

**Expected Impact**:
- **Memory churn**: -95% (3000 allocations → 150)
- **GC pressure**: -90% (minimal garbage)
- **Runtime**: -15-25% for large datasets (n_samples > 1000)
- **Peak memory**: Same (arrays reused)

**Files to Modify**:
- `eclipsera/manifold/_tsne.py`
- New: `eclipsera/manifold/_workspace.py`

---

### 2.2 K-Means Workspace Optimization

#### Current State
```python
# In each iteration:
for iteration in range(self.max_iter):
    distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))  # Allocates
    labels = np.argmin(distances, axis=1)  # Allocates
```

**Problem**: Allocates distance matrix every iteration

#### Proposed Solution
```python
class KMeansWorkspace:
    """Pre-allocated workspace for K-Means computation."""
    
    def __init__(self, n_samples: int, n_clusters: int, n_features: int):
        self.distances = np.zeros((n_samples, n_clusters))
        self.centroids_new = np.zeros((n_clusters, n_features))
        self.counts = np.zeros(n_clusters, dtype=np.int32)
    
    def reset(self):
        self.distances.fill(0)
        self.centroids_new.fill(0)
        self.counts.fill(0)
        return self

# In-place distance computation
def _compute_distances_inplace(X, centroids, distances_out):
    """Compute distances in-place, no allocation."""
    for k in range(centroids.shape[0]):
        diff = X - centroids[k]  # broadcasts, but reusable
        distances_out[:, k] = np.sum(diff * diff, axis=1)
    np.sqrt(distances_out, out=distances_out)
```

**Expected Impact**:
- **Memory churn**: -80% (300 allocations → 60 for max_iter=300)
- **Runtime**: -10-20% for large datasets
- **Scalability**: Better for mini-batch K-Means

**Files to Modify**:
- `eclipsera/cluster/_kmeans.py`
- New: `eclipsera/cluster/_workspace.py`

---

### 2.3 Other Iterative Algorithms

**Candidates for workspace pooling**:

| Algorithm | Current Allocations/Iter | Expected Improvement |
|-----------|-------------------------|---------------------|
| **DBSCAN** | 2-3 (distance matrix, labels) | 5-10% (less iteration-heavy) |
| **Gaussian Mixture** | 5-7 (responsibilities, covariances) | 15-25% |
| **Mean Shift** | 3-4 (distances, weights) | 10-20% |
| **Hierarchical** | 2-3 (linkage updates) | 5-15% |

**Implementation Priority**:
1. t-SNE (highest impact, user-facing)
2. K-Means (high usage, easy wins)
3. Gaussian Mixture (moderate complexity)
4. Others (as needed based on profiling)

---

## Phase 3: Parallel Cross-Validation with ThreadLocalPool

**Target**: v2.1.0  
**Priority**: Medium  
**Estimated Impact**: Near-linear speedup with CPU cores

### 3.1 Parallel GridSearchCV

#### Current State
```python
# Serial execution
for params in param_grid:
    for train_idx, test_idx in cv.split(X, y):
        est.fit(X[train_idx], y[train_idx])  # One at a time
```

**Problem**: Single-threaded, wastes multi-core CPUs

#### Proposed Solution
```python
from joblib import Parallel, delayed
from fastalloc import ThreadLocalPool

class GridSearchCV:
    def __init__(self, ..., n_jobs=-1):
        self.n_jobs = n_jobs  # -1 = all cores
    
    def fit(self, X, y):
        # Create thread-local pools (no locking!)
        if _FASTALLOC_AVAILABLE and self.n_jobs != 1:
            # Each thread gets its own pool
            def create_pool():
                return Pool(
                    obj_type=type(self.estimator),
                    capacity=1,
                    reset_method="reset"
                )
            
            pool = ThreadLocalPool(factory=create_pool)
        
        # Parallel execution
        def evaluate_params(params):
            with pool.allocate() as est:
                scores = []
                for train_idx, test_idx in cv.split(X, y):
                    est.set_params(**params)
                    est.fit(X[train_idx], y[train_idx])
                    scores.append(est.score(X[test_idx], y[test_idx]))
                return np.mean(scores)
        
        # Run in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_params)(params) 
            for params in param_grid
        )
```

**Expected Impact**:
- **Speedup**: ~N×, where N = CPU cores (with GIL release in NumPy)
- **Memory**: Moderate increase (N pools instead of 1)
- **Use case**: Large parameter grids (>100 combinations)

**Key Benefits**:
- No lock contention (ThreadLocalPool)
- Estimator reuse per thread
- Works with existing joblib infrastructure

**Files to Modify**:
- `eclipsera/model_selection/_search.py`
- `eclipsera/model_selection/_split.py` (parallel CV)

---

### 3.2 Thread-Safe Pool Statistics

```python
# Aggregate statistics across all threads
class PoolStatistics:
    def __init__(self):
        self.allocations = 0
        self.hits = 0
        self.misses = 0
        self.resets = 0
    
    def report(self):
        hit_rate = self.hits / (self.hits + self.misses) if self.hits else 0
        print(f"Pool Statistics:")
        print(f"  Allocations: {self.allocations}")
        print(f"  Hit rate: {hit_rate:.1%}")
        print(f"  Resets: {self.resets}")

# Enable with verbose mode
grid = GridSearchCV(..., verbose=2, enable_pool_stats=True)
grid.fit(X, y)
# Automatically prints pool statistics
```

**Impact**: Better observability for tuning

---

## Phase 4: Memory-Efficient Pipeline Optimization

**Target**: v2.2.0  
**Priority**: Low-Medium

### 4.1 Pipeline Buffer Reuse

```python
class Pipeline:
    def __init__(self, steps, reuse_buffers=True):
        self.reuse_buffers = reuse_buffers
        if reuse_buffers:
            # Pre-allocate buffer for intermediate results
            self._buffer_pool = {}
    
    def fit(self, X, y):
        Xt = X
        for name, transformer in self.steps[:-1]:
            if self.reuse_buffers:
                # Reuse buffer for transformed data
                buffer_key = (Xt.shape, Xt.dtype)
                if buffer_key not in self._buffer_pool:
                    self._buffer_pool[buffer_key] = np.empty_like(Xt)
                
                Xt = transformer.fit_transform(Xt, y, out=self._buffer_pool[buffer_key])
            else:
                Xt = transformer.fit_transform(Xt, y)
```

**Impact**: 20-40% less memory for deep pipelines

---

## Phase 5: NumPy Array Pooling (Advanced)

**Target**: v2.3.0  
**Priority**: Low  
**Estimated Impact**: 5-10% additional speedup

### 5.1 Custom Array Allocator

```python
from fastalloc import MemoryPool

# Pool of pre-allocated arrays
array_pool = MemoryPool(
    sizes=[(1000, 10), (5000, 20), (10000, 50)],  # Common shapes
    dtype=np.float64,
    capacity=10
)

def pooled_zeros(shape, dtype=np.float64):
    """Get array from pool or allocate new."""
    return array_pool.get(shape, dtype) or np.zeros(shape, dtype)

# In algorithms:
distances = pooled_zeros((n_samples, n_samples))
# Use distances
array_pool.release(distances)  # Return to pool
```

**Impact**: Marginal for most use cases, significant for tiny repeated fits

---

## Performance Benchmarks (Expected)

### GridSearchCV (10 params × 5 folds = 50 fits)

| Configuration | Time (s) | Speedup | Memory (MB) |
|--------------|----------|---------|-------------|
| v1.0 (no pooling) | 2.50 | 1.0× | 120 |
| v1.2 (pooling) | 2.15 | 1.16× | 118 |
| v2.0 (+ workspace) | 1.90 | 1.32× | 115 |
| v2.1 (+ parallel, 4 cores) | 0.65 | 3.85× | 180 |

### t-SNE (1000 samples, 1000 iterations)

| Configuration | Time (s) | Speedup | Peak Memory (MB) |
|--------------|----------|---------|------------------|
| v1.0 | 45.0 | 1.0× | 850 |
| v2.0 (workspace) | 35.0 | 1.29× | 480 |

### K-Means (10000 samples, 300 iterations)

| Configuration | Time (s) | Speedup | Peak Memory (MB) |
|--------------|----------|---------|------------------|
| v1.0 | 8.2 | 1.0× | 320 |
| v2.0 (workspace) | 6.8 | 1.21× | 280 |

---

## Implementation Timeline

### Q1 2025: Phase 2 - Workspace Pooling
- **Month 1**: t-SNE workspace implementation + testing
- **Month 2**: K-Means workspace implementation + testing
- **Month 3**: Gaussian Mixture + benchmarking

### Q2 2025: Phase 3 - Parallel Execution
- **Month 1**: Parallel GridSearchCV with ThreadLocalPool
- **Month 2**: Parallel cross_val_score/cross_validate
- **Month 3**: Performance profiling + optimization

### Q3 2025: Phase 4 - Pipeline Optimization
- **Month 1**: Pipeline buffer reuse
- **Month 2**: FeatureUnion optimization
- **Month 3**: Integration testing

### Q4 2025: Phase 5 - Advanced Features (Optional)
- **Month 1**: Array pooling evaluation
- **Month 2**: Custom allocator if beneficial
- **Month 3**: Documentation + tutorials

---

## Breaking Changes (Minimal)

### v2.0.0
- **None** - All enhancements are backward compatible
- Workspace pooling only affects internal implementation
- Users can opt-out via environment variable if needed

### v2.1.0
- New parameter: `n_jobs` in GridSearchCV/RandomizedSearchCV (default=1 for compatibility)
- Users must explicitly enable parallelism

---

## Migration Guide

### From v1.x to v2.0
```python
# No changes required!
# Everything works the same, just faster

# Optional: Enable verbose mode to see pooling stats
grid = GridSearchCV(..., verbose=2)
```

### From v2.0 to v2.1 (Parallel)
```python
# Enable parallelism explicitly
grid = GridSearchCV(..., n_jobs=-1)  # Use all cores

# Or specify number of cores
grid = GridSearchCV(..., n_jobs=4)   # Use 4 cores
```

---

## Success Metrics

### Performance Targets
- **GridSearchCV**: 15-20% faster with workspace pooling
- **t-SNE**: 25-30% faster for large datasets
- **Parallel CV**: 3-4× speedup on 4-core machines

### Code Quality
- **Test coverage**: Maintain >90% for new code
- **Backward compatibility**: 100% (no breaking changes)
- **Documentation**: Complete API docs + tutorials

### User Experience
- **Zero-config**: Fast by default, no user changes needed
- **Observable**: Clear logging of pool usage (verbose mode)
- **Stable**: No regressions in accuracy or behavior

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| GIL contention in parallel mode | Medium | Medium | Use ThreadLocalPool, profile carefully |
| Memory overhead from pools | Low | Low | Monitor, cap pool sizes |
| Workspace shape mismatches | Low | High | Extensive testing, runtime validation |
| FastAlloc dependency issues | Low | Medium | Maintain fallback path, vendor if needed |

### User Impact Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Unexpected behavior changes | Very Low | High | Comprehensive regression testing |
| Memory leaks in pooling | Low | High | Valgrind testing, careful resource management |
| Performance degradation | Very Low | Medium | Extensive benchmarking before release |

---

## Community Feedback Integration

### Request for Comments (RFC)
Before implementing Phase 3-5, we will:

1. **RFC Period**: 30 days for community feedback
2. **Prototype**: Proof-of-concept implementation
3. **Benchmarking**: Public benchmark suite
4. **Decision**: Go/no-go based on feedback + data

### Questions for Community
1. Is parallel CV (Phase 3) a priority for your workloads?
2. Which iterative algorithms do you use most?
3. Are there other performance bottlenecks we should address?

---

## Alternative Approaches Considered

### 1. Numba JIT Compilation
**Pros**: Potentially larger speedups  
**Cons**: Complex integration, maintenance burden  
**Decision**: Defer to Phase 6 if needed

### 2. Cython Rewrite of Hot Paths
**Pros**: Proven performance gains  
**Cons**: Increases build complexity  
**Decision**: Consider for Phase 7 if workspace pooling insufficient

### 3. GPU Acceleration
**Pros**: Massive speedups for large problems  
**Cons**: CUDA dependency, complexity  
**Decision**: Out of scope for v2.0, consider v3.0

---

## Conclusion

Eclipsera 2.0.0 will deliver significant performance improvements through intelligent object and workspace pooling, with optional parallel execution in 2.1.0. The focus remains on:

- **Backward compatibility**: Zero breaking changes
- **User experience**: Fast by default, no configuration needed
- **Observability**: Clear insight into performance optimizations
- **Stability**: Extensive testing and gradual rollout

**Expected Overall Impact**: 20-40% faster for typical ML workflows, with potential for 3-4× speedup when parallelism is enabled.

---

## Contributing

Interested in contributing to v2.0? Check out:
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [GitHub Issues: v2.0 Milestone](https://github.com/TIVerse/eclipsera/milestone/ROADMAP_2.0.0.md)
- [Performance Testing Guide](docs/dev/performance_testing.md)

Questions? Open a discussion on GitHub or reach out to the maintainers!

---

**Last Updated**: October 29, 2025  
**Status**: Planning Phase  
**Next Review**: December 2024
