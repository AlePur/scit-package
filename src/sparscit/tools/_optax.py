import jax
from jax import jit
import jax.numpy as jnp
import optax
import numpy as np
from typing import Optional, Literal
from functools import partial
from collections import namedtuple
from jax.scipy.stats import chi2

LogisticParams = namedtuple('LogisticParams', ['coef', 'intercept'])
OptimizerLoopState = namedtuple('LoopState', ['params', 'i', 'grad_norm', 'opt_state'])

@partial(jit, static_argnames=['normalize'])
def log_loss_sklearn(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    normalize: bool = True,
) -> float:
    """Like log_loss from sklearn but JAX.
    
    Parameters
    ----------
    y_true : jnp.ndarray, shape (n_samples,)
        Ground truth labels (must be numeric: 0, 1, 2, ...).
    y_pred : jnp.ndarray, shape (n_samples,)
        Predicted probabilities (from the positive class).
    normalize : bool, default=True
        If true, return the mean loss per sample.
        
    Returns
    -------
    loss : float
        Log loss.
    """
    eps = jnp.finfo(y_pred.dtype).eps
    
    # Handle binary case (1D predictions)
    assert y_pred.ndim == 1
    y_pred = jnp.clip(y_pred, eps, 1 - eps)
    loss = -(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

    return jnp.mean(loss) if normalize else jnp.sum(loss)

@partial(jit, static_argnames=['normalize'])
def _log_loss(params: LogisticParams, X: jnp.ndarray, y: jnp.ndarray, normalize: bool = True):
    """
    Compute binary cross-entropy loss for logistic regression.

    Parameters
    ----------
    params : LogisticParams
        Named tuple of (coef, intercept)
    X : jnp.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : jnp.ndarray
        Binary target labels of shape (n_samples,)
    normalize : bool
        If True, return mean loss per sample; otherwise return sum

    Returns
    -------
    float
        Binary cross-entropy loss value
    """
    w, b = params
    logits = jnp.dot(X, w) + b
    # Binary cross-entropy
    log_probs = jax.nn.log_sigmoid(logits)
    log_probs_neg = jax.nn.log_sigmoid(-logits)
    if normalize:
        loss = -jnp.mean(y * log_probs + (1 - y) * log_probs_neg)
    else:
        loss = -jnp.sum(y * log_probs + (1 - y) * log_probs_neg)
    return loss

@partial(jit, static_argnames=['max_iter', 'tol'])
def fit_log_regression(
        X: jnp.ndarray, 
        y: jnp.ndarray,
        *,
        max_iter: int = 100,
        tol: float = 1e-5,
        warm_params: LogisticParams = LogisticParams(coef = jnp.array([0]), intercept=jnp.array(0.0))
) -> LogisticParams:
    """
    Fit a binary logistic regression model using L-BFGS via Optax.

    Parameters
    ----------
    X : jnp.ndarray, shape (n_samples, n_features)
        Training data
    y : jnp.ndarray, shape (n_samples,)
        Target values (binary: 0 or 1)
    max_iter
        Maximum number of optimization iterations
    tol
        Convergence tolerance on gradient norm
    warm_params : LogisticParams
        Initial parameters. Must have the correct shape for X.
        If unsure, set coef and intercept to zero arrays of appropriate shape.

    Returns
    -------
    LogisticParams
        Named tuple with fitted ``coef`` and ``intercept``
    """
    print('Debug: traced fit_log_regression')
    # Initialize parameters
    # w_init = jax.random.normal(key, (X.shape[1],)) * 0.01
    # w_init = jnp.where(warm_params.coef == 0, w_init, warm_params.coef)
    w_init = warm_params.coef
    b_init = warm_params.intercept
    #b_init = jnp.array(0.0)
    params = (w_init, b_init)
    
    # Create optimizer
    optimizer = optax.lbfgs(
        learning_rate=None,
        memory_size=20,
        scale_init_precond=True,
    )
    
    # Define value and gradient function
    value_and_grad_fn = jax.value_and_grad(_log_loss)

    def cond_f(loop_state: OptimizerLoopState):
        return (loop_state.i < max_iter) * (loop_state.grad_norm > tol)
    
    # Optimization loop
    def body_f(loop_state: OptimizerLoopState) -> OptimizerLoopState:
        loss, grads = value_and_grad_fn(loop_state.params, X, y)
        
        # Update parameters
        updates, new_opt_state = optimizer.update(
            grads, loop_state.opt_state, loop_state.params, value=loss, grad=grads, value_fn=lambda p: _log_loss(p, X, y)
        )
        new_params = optax.apply_updates(loop_state.params, updates)
        
        # Check convergence
        new_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))

        return OptimizerLoopState(
            new_params, loop_state.i + 1, new_grad_norm, new_opt_state
        )
        
    final_params, loop_i, _, _ = jax.lax.while_loop(cond_f, body_f, OptimizerLoopState(
        params, 0, 10000, optimizer.init(params)
    ))
    # jax.debug.print("{x}", x=loop_i)
    return LogisticParams(final_params[0], final_params[1])

@jit
def predict_proba(params: LogisticParams, X: jnp.ndarray) -> jnp.ndarray:
    """Predict class probabilities.
    
    Parameters
    ----------
    params : LogisticParams
        Named tuple of (coef, intercept) with fitted logistic regression parameters
    X : jnp.ndarray, shape (n_samples, n_features)
        Samples.
        
    Returns
    -------
    proba : jnp.ndarray, shape (n_samples, 2)
        Probability of each class (class 0, class 1).
    """
    X = jnp.array(X, dtype=jnp.float32)
    logits = jnp.dot(X, params.coef.T).squeeze() + params.intercept
    prob_class1 = jax.nn.sigmoid(logits)
    prob_class0 = 1 - prob_class1
    
    return jnp.column_stack([prob_class0, prob_class1])

@partial(jit, static_argnames=['ddof'])
def likelihood_ratio_test(
        X0: jnp.ndarray,
        X1: jnp.ndarray,
        y: jnp.ndarray,
        reduced_params: LogisticParams,
        *,
        ddof: int = 1
) -> jnp.ndarray:
    """
    Adopted from SnapATAC2, added JAX implementations.
    Comparing null model with alternative model using the likelihood ratio test.
    reduced_params needs to be correct for the X and y, otherwise the result will not be sensible.

    Parameters
    ----------
    X0
        (n_sample, n_feature), variables used in null model.
    X1
        (n_sample, n_feature), variables used in alternative model.
        Note: X1 does NOT contain X0, it's added to it when this function runs.
    y
        (n_sample, ), labels.
    reduced_params : LogisticParams
        Pre-fitted parameters for the reduced (null) model on X0.
    ddof : int
        Degrees of freedom for the chi-squared test (default 1).

    Returns
    -------
    jnp.ndarray
        The P-value from the likelihood ratio test.
    """
    print('Debug: traced likelihood_ratio_test')
    # s: LogisticParams = fit_log_regression(X0, y, warm_params=warm_params)
    reduced = -_log_loss(reduced_params, X0, y, normalize=False)

    X_both = jnp.c_[X0, X1]

    # it is different shape
    s = LogisticParams(coef=jnp.array([reduced_params.coef[0], reduced_params.coef[0]]), intercept=reduced_params.intercept)
    s = fit_log_regression(X_both, y, warm_params=s)
    full = -_log_loss(s, X_both, y, normalize=False)

    chi = -2 * (reduced - full)
    return chi2.sf(chi, ddof) # X_both.shape[1] - X0.shape[1]