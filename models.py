from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.sequential_stack = nn.Sequential(
            nn.Linear(input_dim, 34),
            nn.ReLU(),
            nn.Linear(34, 17),
            nn.ReLU(),
            nn.Linear(17, output_dim),
        )

    def forward(self, x):
        output = self.sequential_stack(x)
        return output