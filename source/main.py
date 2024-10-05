import CHopfield
import torch


def main(temperature: float, n_neurons: int, n_patterns: int, n_iter: int):
    hopfield = CHopfield.CHopfield(temperature, n_patterns, n_neurons, n_iter=n_iter)
    x = torch.randn(9, n_neurons)
    print(hopfield(x))
    print(hopfield.patterns)


if __name__ == "__main__":
    temperature = 0.5
    n_neurons = 8
    n_patterns = 2
    n_iter = 8
    main(
        temperature=temperature,
        n_neurons=n_neurons,
        n_patterns=n_patterns,
        n_iter=n_iter,
    )
