import torch
from torch import nn


class HopfieldLayer(nn.Module):

    _n_classes: int
    _n_neurons: int
    _n_features: int
    n_iter: int
    beta: float

    _patterns: torch.Tensor  # (n_features, n_classes, n_neurons)

    _classifier: bool  # if True, the layer is used as a classifier with logits, otherwise it is used as a feature extractor

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_neurons: int,
        n_iter: int,
        temperature: float,
        classifier: bool = True,
    ):
        """Initialize the Hopfield layer.

        Args:
            n_features (int): Num of features
            n_classes (int): Num of classes
            n_neurons (int): Num of neurons
            n_iter (int): Num of iterations
            temperature (float): Temperature
            classifier (bool): If True, the layer is used as a classifier with logits, otherwise it is used as a feature extractor (default: True)
        """
        super(HopfieldLayer, self).__init__()

        # Check input parameters
        assert n_features > 0, "n_features must be greater than 0"
        assert n_classes > 0, "n_classes must be greater than 0"
        assert n_neurons > 0, "n_neurons must be greater than 0"
        assert n_iter >= 0, "n_iter must be greater than 0"
        assert temperature > 0, "temperature must be greater than 0"

        # Algorithm
        self._n_features = n_features
        self._n_classes = n_classes
        self._n_neurons = n_neurons
        self.n_iter = n_iter
        self.beta = 1 / temperature
        self._patterns = nn.Parameter(
            torch.tanh(torch.randn(n_features, n_classes, n_neurons))
        )
        self._classifier = classifier

    def store(self, patterns: torch.Tensor):
        """Train the Hopfield layer.

        Args:
            patterns (torch.Tensor): Patterns with shape (n_features, n_classes, n_neurons)
        """
        assert (
            patterns.shape[0] == self._n_features
        ), "patterns must have the same number of features as the layer"
        assert (
            patterns.shape[1] == self._n_classes
        ), "patterns must have the same number of classes as the layer"
        assert (
            patterns.shape[2] == self._n_neurons
        ), "patterns must have the same number of neurons as the layer"

        self._patterns.data = patterns

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Hopfield layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, n_features, n_neurons)

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, n_features, n_classes) if classifier is True, otherwise (batch_size, n_features, n_neurons)
        """

        # Check input parameters
        try:
            x = torch.reshape(x, (x.shape[0], self._n_features, self._n_neurons))
        except Exception:
            raise ValueError(
                "x must have the same number of features and neurons as the layer"
            )

        # Algorithm
        A = self.beta * torch.einsum(
            "fin,fjn->fij", self._patterns, self._patterns
        )  # (n_features, n_classes, n_classes)
        logits = self.beta * torch.einsum(
            "fcn, bfn -> bfc", self._patterns, x
        )  # (batch_size, n_features, n_classes)

        for _ in range(self.n_iter):
            logits = torch.einsum(
                "fck, bfk -> bfc", A, torch.nn.functional.softmax(logits, dim=2)
            )

        if self._classifier:
            return logits
        else:
            return torch.einsum(
                "fcn, bfc -> bfn",
                self._patterns,
                torch.nn.functional.softmax(logits, dim=2),
            )
