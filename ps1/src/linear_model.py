"""Base class shared by the PS1 linear models."""


class LinearModel:
    """Abstract base for a linear model parameterised by ``theta``."""

    def __init__(self, step_size=0.2, max_iter=100, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative (gradient) solvers.
            max_iter: Maximum number of solver iterations.
            eps: Convergence threshold on the parameter update.
            theta_0: Initial parameter vector (defaults to zeros).
            verbose: Whether solvers should print progress.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
