r"""
Instantiation helpers for a MLP with support for
- Goodfellow's trick for computing individual gradients
- Parrallel computation of multiple parameters (e.g. for additive noise)
"""

import torch
from torch.nn import Module
from torch.nn.modules.linear import Linear
import torchutils.params as paramutils

__all__ = [
    'MLP'
]

class LinearWithSampling(Linear):
    r"""Extension of the ``torch.nn.Linear`` Module with support for sampling.

    See :meth:`torch.nn.Linear`` for a full documentation.
    """
    def __init__(self, in_features, out_features):
        super(LinearWithSampling, self).__init__(in_features, out_features, True)

    def forward(self, x, weight_noise=None, bias_noise=None):
        assert (
            ((weight_noise is None) and (bias_noise is None)) or
            ((weight_noise is not None) and (bias_noise is not None))
        )

        if weight_noise is None:
            output = x.matmul(self.weight.t())
            output += self.bias
        else:
            output = torch.bmm(
                x,
                (self.weight.unsqueeze(0).expand(x.shape[0], -1, -1) + weight_noise).transpose(1, 2)
            )
            output += self.bias + bias_noise.unsqueeze(1).expand(-1, output.shape[1], -1)

        return output

class MLP(Module):
    r"""MLP with additional support for individual gradients and sampling.

    Additional capabilities:

        * Sampling with additive noise to the parameters
        * Individual Gradients computation

    Sampling:
        Let D be the number of parameters of the MLP.
        Forward accepts a `noise` parameter, a `[D x S]` matrix
        representing `S` independent samples of additive noise.

        The ordering of the parameters follows the conventions of
        * `torch.nn.utils.parameters_to_vector`
        * `torch.nn.utils.vector_to_parameters`

    Individual gradients computations:
        To support manual differentiation of each layer,
        the `forward` pass accepts a `indgrad` parameter

    Example:
        Creates a MLP with two hidden layers of size [64, 16],
        taking 256-valued input and returning a single output.

            >>> model = MLP(256, [64, 16], 1)

    Arguments:
        input_size (int): Size of the input.
        hidden_sizes (List of int): Size of the hidden layers.
            Defaults to [] (no hidden layer).
        output_size (int): Size of the output.
            Defaults to 1
        act_func: Activation function (see ``torch.nn.functional``).
            Defaults to ``torch.tanh``.
    """
    def __init__(self, input_size, hidden_sizes=None, output_size=1, act_func=torch.tanh):
        super(MLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = []

        self.input_size = input_size
        self.output_size = output_size
        self.act = act_func

        if len(hidden_sizes) == 0:
            self.hidden_layers = []
            self.output_layer = LinearWithSampling(self.input_size, self.output_size)
        else:
            self.hidden_layers = torch.nn.ModuleList([
                LinearWithSampling(in_size, out_size) for in_size, out_size in
                zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            ])
            self.output_layer = LinearWithSampling(hidden_sizes[-1], self.output_size)

    def forward(self, x, noise=None, indgrad=False):
        r"""Forward pass with support for additive noise to the parameters.

        :code:`x` needs to be a [N x input_size] matrix, where N is the
        minibatch dimension, and
        :code:`noise` needs to be None or a [S x D] dimension, where S is the
        number of samples and D matches the number of parameters of the model.

        **Sampling:**

        Batch computation with additive noise to the parameters
        is supported through the :code:`noise` argument, a [S x D] Tensor
        representing additive noise to the parameters.

        **Individual gradients computations:**

        To support manual differentiation of each layer,
        the `forward` function returns the inputs and output
        of each linear layer in parameter format.

        See :mod:`curvfuncs` for use cases.

        Arguments:
            x (Tensor): [N x input_size]
            noise (Tensor): [S x D] additive noise matrix matrix,
                where `D` is the number of model parameters.
                Defaults to a no additive noise.

        Returns:
            A tuple containing

                * **y** (Tensor): [(S) x N x output_size] output, where
                  S is the noise batch dimension and N is the minibatch
                  dimension.
                  The noise dimension is not present if no noise was passed,
                  and thus a [N x output_size] matrix is returned instead.
                * **activations**, **linear_combs** (Optional - if ``indgrad`` is ``True``):
                  list of the input and output tensors of each layer
                  to manually compute individual gradients.
        """

        x = x.view([-1, self.input_size])

        if noise is not None and isinstance(noise, torch.Tensor):
            assert len(noise.shape) == 2
            assert noise.shape[1] == paramutils.num_params(self.parameters())
            noise = paramutils.bv2p(noise, self.parameters())

        if noise is None:
            activation = x
        else:
            activation = x.expand(noise[0].shape[0], -1, -1)

        if indgrad:
            activations = [activation]
            linear_combs = []

        for layer_id in range(len(self.hidden_layers)):
            if noise is None:
                linear_comb = self.hidden_layers[layer_id](activation)
            else:
                linear_comb = self.hidden_layers[layer_id](
                    activation, noise[2*layer_id], noise[2*layer_id + 1]
                )

            activation = self.act(linear_comb)

            if indgrad:
                linear_combs.append(linear_comb)
                activations.append(activation)

        if noise is None:
            output = self.output_layer(activation)
        else:
            output = self.output_layer(activation, noise[-2], noise[-1])

        if indgrad:
            linear_combs.append(output)

        if indgrad:
            return output, activations, linear_combs
        return output

def num_params(model):
    r"""
    Returns the number of parameters registered in ``model``.
    """
    return sum([p.numel() for p in model.parameters()])
