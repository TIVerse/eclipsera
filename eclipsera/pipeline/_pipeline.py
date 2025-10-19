"""Pipeline of transforms with a final estimator."""
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from ..core.base import BaseEstimator
from ..core.validation import check_array


class Pipeline(BaseEstimator):
    """Pipeline of transforms with a final estimator.
    
    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.
    
    Parameters
    ----------
    steps : list of tuples
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed.
        
    Attributes
    ----------
    named_steps : dict
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        
    Examples
    --------
    >>> from eclipsera.pipeline import Pipeline
    >>> from eclipsera.preprocessing import StandardScaler
    >>> from eclipsera.ml import LogisticRegression
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('clf', LogisticRegression())
    ... ])
    >>> X = [[0, 0], [1, 1]]
    >>> y = [0, 1]
    >>> pipe.fit(X, y)
    Pipeline(...)
    >>> pipe.predict([[0.5, 0.5]])
    array([0])
    """
    
    def __init__(self, steps: List[Tuple[str, Any]], verbose: bool = False):
        self.steps = steps
        self.verbose = verbose
        self._validate_steps()
    
    def _validate_steps(self):
        """Validate the steps."""
        if not self.steps:
            raise ValueError("Pipeline must have at least one step")
        
        # Check that all but last implement fit and transform
        for idx, (name, estimator) in enumerate(self.steps[:-1]):
            if not (hasattr(estimator, 'fit') and hasattr(estimator, 'transform')):
                raise TypeError(
                    f"All intermediate steps should implement fit and transform. "
                    f"'{name}' (type {type(estimator)}) doesn't"
                )
        
        # Check that last implements fit
        if not hasattr(self.steps[-1][1], 'fit'):
            raise TypeError(
                f"Last step should implement fit. "
                f"'{self.steps[-1][0]}' (type {type(self.steps[-1][1])}) doesn't"
            )
    
    @property
    def named_steps(self) -> dict:
        """Access the steps by name."""
        return dict(self.steps)
    
    def _iter(self, with_final: bool = True):
        """Generate (name, transform) tuples."""
        if with_final:
            yield from self.steps
        else:
            yield from self.steps[:-1]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> "Pipeline":
        """Fit the model.
        
        Fit all the transforms one after the other and transform the data,
        then fit the transformed data using the final estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values.
        **fit_params : dict
            Parameters passed to the fit method of each step.
            
        Returns
        -------
        self : Pipeline
            This estimator.
        """
        X = check_array(X)
        if y is not None:
            y = check_array(y, ensure_2d=False)
        
        Xt = X
        for step_idx, (name, transformer) in enumerate(self._iter(with_final=False)):
            if self.verbose:
                print(f"[Pipeline] Fitting {name}")
            
            # Fit and transform
            Xt = transformer.fit_transform(Xt, y)
        
        # Fit final estimator
        if self.verbose:
            print(f"[Pipeline] Fitting {self.steps[-1][0]}")
        
        self.steps[-1][1].fit(Xt, y)
        
        return self
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> np.ndarray:
        """Fit all transforms and transform the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values.
        **fit_params : dict
            Parameters passed to the fit method of each step.
            
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        X = check_array(X)
        if y is not None:
            y = check_array(y, ensure_2d=False)
        
        Xt = X
        for name, transformer in self._iter(with_final=True):
            if self.verbose:
                print(f"[Pipeline] Fitting and transforming {name}")
            
            Xt = transformer.fit_transform(Xt, y)
        
        return Xt
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply transforms and predict with final estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.
            
        Returns
        -------
        y_pred : ndarray
            Predictions.
        """
        X = check_array(X)
        
        Xt = X
        for name, transformer in self._iter(with_final=False):
            Xt = transformer.transform(Xt)
        
        return self.steps[-1][1].predict(Xt)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply transforms to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        Xt : ndarray
            Transformed data.
        """
        X = check_array(X)
        
        Xt = X
        for name, transformer in self._iter(with_final=False):
            Xt = transformer.transform(Xt)
        
        # Also transform with final step if it has transform
        if hasattr(self.steps[-1][1], 'transform'):
            Xt = self.steps[-1][1].transform(Xt)
        
        return Xt
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Apply transforms and score with final estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score on.
        y : array-like of shape (n_samples,)
            True labels.
            
        Returns
        -------
        score : float
            Score of the final estimator.
        """
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        
        Xt = X
        for name, transformer in self._iter(with_final=False):
            Xt = transformer.transform(Xt)
        
        return self.steps[-1][1].score(Xt, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Apply transforms and predict_proba with final estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.
            
        Returns
        -------
        y_proba : ndarray
            Class probabilities.
        """
        X = check_array(X)
        
        Xt = X
        for name, transformer in self._iter(with_final=False):
            Xt = transformer.transform(Xt)
        
        return self.steps[-1][1].predict_proba(Xt)
    
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        if not deep:
            return {'steps': self.steps, 'verbose': self.verbose}
        
        out = {'steps': self.steps, 'verbose': self.verbose}
        
        for name, estimator in self.steps:
            for key, value in estimator.get_params(deep=True).items():
                out[f'{name}__{key}'] = value
        
        return out
    
    def set_params(self, **params) -> "Pipeline":
        """Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : Pipeline
            Pipeline instance.
        """
        if not params:
            return self
        
        # Handle step-specific parameters
        step_params = {}
        for key, value in params.items():
            if '__' in key:
                step, param = key.split('__', 1)
                if step not in step_params:
                    step_params[step] = {}
                step_params[step][param] = value
            else:
                setattr(self, key, value)
        
        # Set parameters for each step
        for step_name, step_param_dict in step_params.items():
            for name, estimator in self.steps:
                if name == step_name:
                    estimator.set_params(**step_param_dict)
                    break
        
        return self


class FeatureUnion(BaseEstimator):
    """Concatenates results of multiple transformer objects.
    
    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.
    
    Parameters
    ----------
    transformer_list : list of tuples
        List of (name, transformer) tuples that are applied to the data.
    n_jobs : int, default=None
        Number of jobs to run in parallel (not implemented yet).
    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be printed.
        
    Examples
    --------
    >>> from eclipsera.pipeline import FeatureUnion
    >>> from eclipsera.preprocessing import StandardScaler, MinMaxScaler
    >>> union = FeatureUnion([
    ...     ('scaler1', StandardScaler()),
    ...     ('scaler2', MinMaxScaler())
    ... ])
    >>> X = [[0, 1], [1, 0]]
    >>> union.fit_transform(X)
    array([...])
    """
    
    def __init__(
        self,
        transformer_list: List[Tuple[str, Any]],
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureUnion":
        """Fit all transformers.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,), default=None
            Target values.
            
        Returns
        -------
        self : FeatureUnion
            This estimator.
        """
        X = check_array(X)
        if y is not None:
            y = check_array(y, ensure_2d=False)
        
        for name, transformer in self.transformer_list:
            if self.verbose:
                print(f"[FeatureUnion] Fitting {name}")
            transformer.fit(X, y)
        
        return self
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit all transformers and concatenate results.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,), default=None
            Target values.
            
        Returns
        -------
        X_transformed : ndarray
            Horizontally stacked results of transformers.
        """
        X = check_array(X)
        if y is not None:
            y = check_array(y, ensure_2d=False)
        
        results = []
        for name, transformer in self.transformer_list:
            if self.verbose:
                print(f"[FeatureUnion] Fitting and transforming {name}")
            Xt = transformer.fit_transform(X, y)
            results.append(Xt)
        
        return np.hstack(results)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X separately by each transformer and concatenate results.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        X_transformed : ndarray
            Horizontally stacked results of transformers.
        """
        X = check_array(X)
        
        results = []
        for name, transformer in self.transformer_list:
            Xt = transformer.transform(X)
            results.append(Xt)
        
        return np.hstack(results)
    
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        if not deep:
            return {
                'transformer_list': self.transformer_list,
                'n_jobs': self.n_jobs,
                'verbose': self.verbose
            }
        
        out = {
            'transformer_list': self.transformer_list,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }
        
        for name, transformer in self.transformer_list:
            for key, value in transformer.get_params(deep=True).items():
                out[f'{name}__{key}'] = value
        
        return out


def make_pipeline(*steps, verbose: bool = False) -> Pipeline:
    """Construct a Pipeline from the given estimators.
    
    This is a shorthand for the Pipeline constructor; it does not require,
    and does not permit, naming the estimators. Instead, their names will
    be set to the lowercase of their types automatically.
    
    Parameters
    ----------
    *steps : list of estimators
        List of estimators.
    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed.
        
    Returns
    -------
    pipeline : Pipeline
        Returns a scikit-learn Pipeline object.
        
    Examples
    --------
    >>> from eclipsera.pipeline import make_pipeline
    >>> from eclipsera.preprocessing import StandardScaler
    >>> from eclipsera.ml import LogisticRegression
    >>> pipe = make_pipeline(StandardScaler(), LogisticRegression())
    """
    names = [type(estimator).__name__.lower() for estimator in steps]
    
    # Make names unique
    name_counts = {}
    for idx, name in enumerate(names):
        if name in name_counts:
            name_counts[name] += 1
            names[idx] = f"{name}_{name_counts[name]}"
        else:
            name_counts[name] = 0
    
    return Pipeline(list(zip(names, steps)), verbose=verbose)
