r"""
Handling of parameter formats and other parameter-related utilities.

Parameter Formats
^^^^^^^^^^^^^^^^^

Batch versions of the functions :meth:`vector_to_parameters`
and :meth:`parameters_to_vector` of the :mod:`torch.nn.utils` module
allows to go from PyTorch's Parameter Format to a single tensor format.

Example:

    Consider network consisting of two linear layers where
    the input, hidden and output dimensions are D, H and O.

    * The ``Parameter Format (PF)``, given by :meth:`model.parameters`,
      list tensors for each layer and
      would contain a [D x H] and a [H x O] tensor.
    * The ``vector format`` would be a single [(D*H*O) x 1] tensor.

Batch Parameter Formats
^^^^^^^^^^^^^^^^^^^^^^^

The batch version allows to specify batch dimensions to convert multiple
sets of parameters between those two formats.

    * :meth:`params.bv2p`: A tensor of shape [S x D], where D matches
      the number of parameter a given :meth:`model.parameters`,
      can be converted to a list of tensors of shapes [S x ...]
      where ``...`` matches the shape of each tensor of
      the :meth:`model.parameters`
    * :meth:`params.bp2v`: A list of batch parameter format in batch form,
      where each tensor has some batch dimension [S x ...] will have its
      non batch-dimensions ``...`` flattened and returned as a single
      [S x D] tensor, where D is the number of parameters

Both methods support an arbitrary number of batch dimensions.

.. note::

    :meth:`torch.nn.utils.vector_to_parameters` (abbreviated ``v2p``) is an
    in-place function which takes the vector and _assigns_ it to the parameters.
    The batch version ``bv2p`` here takes a vector and _returns_ the list in
    parameter format.
    Passing a one dimensional tensor to :meth:`params.bv2p` is an out-of-place
    version of :meth:`torch.nn.utils.vector_to_parameters`.


"""

import torch

__all__ = [
    "num_params",
    "batch_parameterformat_to_vec", "bp2v",
    "batch_vec_to_parameterformat", "bv2p",
]

def num_params(parameters_pf):
    r"""
    Returns the number of parameters in a list of parameters (pf)
    """
    return sum([param.numel() for param in parameters_pf])

def batch_vec_to_parameterformat(params, parameters_pf):
    r"""
    Batch conversion from vector format to parameter format

    .. note::

        This method does **not** modifies the content of ``parameters_pf``
        but simply returns the ``params`` in a list-of-tensor

    **Example: 1 Batch dimension:**

    Given a tensor ``params`` of shape ``[S x D]``
    and a matching parameter list ``parameters_pf`` list
    containing two tensors of sizes ``[a x b]`` and ``[c x 1]``
    (such that :math:`ab + c = D`), returns
    a list of tensors with sizes ``[S x a x b], [S x c x 1]``.

    **Example: 2 Batch dimensions:**

    Given a tensor ``params`` of shape ``[A x B x D]``
    and a matching parameter list ``parameters_pf`` list
    containing two tensors of sizes ``[a x b]`` and ``[c x d]``
    (such that :math:`ab + cd = D`), returns
    a list of tensors with sizes ``[A x B x a x b], [A x B x c x d]``.

    Arguments:
        params (Tensor): [... x D] where D is the number of parameters
            of the model and ``...`` are batch dimensions.

    Returns:
        **params_pf** (List of Tensor) The length of the list will match
        the length of ``parameters_pf`` and the tensor at index ``i``
        will be of size ``[... x ,,,]``, where ``...`` are the batch dimensions
        and ``,,,`` are the dimensions of ``parameters_pf[i]``
        (See Example)
    """
    parameters_pf = list(parameters_pf)

    assert params.shape[-1] == num_params(parameters_pf)

    param_format = []
    pointer = 0
    for param in parameters_pf:
        num_param = param.numel()

        param_format.append(
            params.narrow(-1, pointer, num_param).view(
                *params.shape[:-1], *param.shape
            )
        )

        pointer += num_param
    return param_format

def batch_parameterformat_to_vec(params_pf, batch_dims):
    r"""
    Batch conversion from parameter format to vector format

    Inverse of :meth:`params.batch_vec_to_parameterformat`

    Arguments:
        params_pf (List of Tensor): parameters in parameter format
        batch_dims (int): number of batch dimensions

    Returns:
        A Tensor in Matrix Form with the batch dimensions conserved
    """
    params_reformatted = []
    for param in params_pf:
        newshape = list(param.shape[0:batch_dims]) + [-1]
        params_reformatted.append(param.view(newshape))
    return torch.cat(params_reformatted, dim=-1)

def bp2v(params_pf, batch_dims):
    r"""Shortcut for :meth:`params.batch_parameterformat_to_vec`
    """
    return batch_parameterformat_to_vec(params_pf, batch_dims)

def bv2p(params, params_pf):
    r"""Shortcut for :meth:`params.batch_vec_to_parameterformat`
    """
    return batch_vec_to_parameterformat(params, params_pf)

def flatten_last_dims(params_pf, dims):
    r"""Flattens the last ``dims`` dimensions of each tensor in ``params_pf``
    and concatenate along the resulting flattened dimension.
    """
    mats = []
    for param in params_pf:
        newshape = list(param.shape)[:-(dims+1)] + [-1]
        mats.append(param.view(newshape))
    return torch.cat(mats, dim=-1)
    