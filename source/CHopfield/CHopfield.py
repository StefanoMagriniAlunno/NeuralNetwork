import torch


class CHopfield(torch.nn.Module):

    temperature: float  # temperature of the system
    n_patterns: int  # number of memorized patterns
    n_neurons: int  # number of neurons
    n_iter: int  # max number of iterations
    patterns: torch.Tensor  # memorized patterns
    coefficient: torch.Tensor  # maximum norm of the patterns

    def __init__(
        self,
        temperature: float,
        n_patterns: int,
        n_neurons: int,
        initial_variance: float = 0.1,
        n_iter: int = 1,
    ):
        super(CHopfield, self).__init__()
        assert temperature > 0, "Temperature must be positive"
        assert n_patterns > 0, "Number of patterns must be positive"
        assert n_neurons > 0, "Number of neurons must be positive"
        assert initial_variance >= 0, "Initial variance must be positive"
        assert n_iter > 0, "Number of iterations must be positive"

        self.temperature = temperature
        self.n_patterns = n_patterns
        self.n_neurons = n_neurons
        self.n_iter = n_iter

        self.patterns = torch.tanh(
            torch.randn(n_patterns, n_neurons, dtype=torch.float32, device="cpu")
            * initial_variance
        )
        self.coefficient = torch.exp(
            -0.5
            * self.temperature
            * (
                torch.max(torch.norm(self.patterns, p=2, dim=1)) ** 2
                - torch.norm(self.patterns, p=2, dim=1) ** 2
            )
        )

    def train(self, patterns: torch.Tensor):
        assert (
            patterns.shape[0] == self.n_patterns
        ), "Number of patterns must be equal to the number of memorized patterns"
        assert (
            patterns.numel() == self.n_patterns * self.n_neurons
        ), "Patterns must have the same number of neurons as the model"

        self.patterns = (
            patterns.clone().detach().reshape(self.n_patterns, self.n_neurons)
        )
        self.coefficient = torch.exp(
            -0.5
            * self.temperature
            * (
                torch.max(torch.norm(self.patterns, p=2, dim=1)) ** 2
                - torch.norm(self.patterns, p=2, dim=1) ** 2
            )
        )

    def energy(self, x: torch.Tensor):
        assert len(x.shape) == 2, "Input must be a 2D tensor"
        assert (
            x.shape[1] == self.n_neurons
        ), "Input must have the same number of neurons as the model"
        assert x.dtype == torch.float32, "Input must be of type float32"

        return (
            -torch.log(
                torch.mean(
                    torch.exp(
                        -0.5
                        * self.temperature
                        * torch.norm(
                            x[:, None, :] - self.patterns[None, :, :], p=2, dim=2
                        )
                        ** 2
                    )
                    * self.coefficient[None, :],
                    dim=1,
                )
            )
            / self.temperature
        )

    def fix(self, trainable: bool = False):
        self.patterns.requires_grad = trainable
        self.coefficient.requires_grad = trainable

    def step(self, x: torch.Tensor):
        assert len(x.shape) == 2, "Input must be a 2D tensor"
        assert (
            x.shape[1] == self.n_neurons
        ), "Input must have the same number of neurons as the model"
        assert x.dtype == torch.float32, "Input must be of type float32"

        return (
            torch.softmax(self.temperature * x @ self.patterns.T, dim=1) @ self.patterns
        )

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 2, "Input must be a 2D tensor"
        assert (
            x.shape[1] == self.n_neurons
        ), "Input must have the same number of neurons as the model"
        assert x.dtype == torch.float32, "Input must be of type float32"

        x = x.clone().detach().reshape(-1, self.n_neurons)
        for _ in range(self.n_iter):
            x = self.step(x)
        return x
