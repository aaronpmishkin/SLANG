# @Author: amishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   amishkin
# @Last modified time: 18-08-17



class LowRankBayesianMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act_func, weight_prior_prec=1.0, bias_prior_prec=1.0, L=1, use_cuda=torch.cuda.is_available()):
        super(LRDNN, self).__init__()
        self.input_size = input_size
        self.use_cuda = use_cuda
        self.L = L
        self.hidden_sizes = hidden_sizes
        if output_size:
            self.output_size = output_size
            self.squeeze_output = False
        else :
            self.output_size = 1
            self.squeeze_output = True
        self.act = F.tanh if act_func == "tanh" else F.relu
        if len(hidden_sizes) == 0:
            self.hidden_layers = []
            self.output_layer = MeanLinear(self.input_size, self.output_size)
        else:
            self.hidden_layers = nn.ModuleList([MeanLinear(in_size, out_size) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            self.output_layer = MeanLinear(hidden_sizes[-1], self.output_size)


        # At this point, the parameters should only be the layer means.
        params = self.parameters()
        param_vec = parameters_to_vector(params)
        self.diagonal = Parameter(torch.empty_like(param_vec), requires_grad=True)
        self.V = Parameter(torch.empty(param_vec.size()[0], L), requires_grad=True)
        self.prior_prec = self.construct_prior_vector(weight_prior_prec, bias_prior_prec, self.named_parameters())
        self.reset_model()


    def reset_model(self):
        self.diagonal.detach().fill_(1)
        self.V.detach().normal_()

        for layer in self.hidden_layers:
            layer.reset_parameters()
        self.output_layer.reset_parameters()

    def get_mu_params(self):
        parameters = self.named_parameters()
        params = []

        # Obtain **only** the network (i.e mean) parameters
        for i,p in enumerate(parameters):
            if p[0] != 'diagonal' and p[0] != 'V':
                params.append(p[1])

        return params

    def forward(self, x):
        raw_noise = torch.normal(mean=torch.zeros_like(self.diagonal), std=1.0)
        # raw_noise = torch.zeros_like(self.diagonal)
        Sigma = torch.mm(self.V, self.V.t()) + torch.diag(F.softplus(self.diagonal))
        U = torch.potrf(Sigma)
        cov_noise = torch.mv(U.t(), raw_noise)

        # Feed forward through the network.
        x = x.view(-1,self.input_size)
        out = x
        offset = 0
        sizes = get_block_dimensions(self.input_size, self.output_size, self.hidden_sizes)
        for i, layer in enumerate(self.hidden_layers):
            noise_vec = cov_noise[offset:sizes[i]+offset]
            out = self.act(layer(out, noise_vec))
            offset += sizes[i]
        logits = self.output_layer(out, cov_noise[offset:])

        if self.squeeze_output:
            logits = torch.squeeze(logits)

        return logits


    def construct_prior_vector(self, weight_prior, bias_prior, named_parameters):
        prior_list = []
        offest = 0
        for i,p in enumerate(named_parameters):
            if 'bias' in p[0]:
                prior_list.append(torch.ones_like(p[1]).mul(bias_prior))
            elif p[0] != 'diagonal' and p[0] != 'V':
                prior_list.append(torch.ones_like(p[1]).mul(weight_prior))


        prior_vector = parameters_to_vector(prior_list)
        if use_cuda:
            prior_vector = prior_vector.cuda()

        return prior_vector


    def kl_divergence(self):
        '''
            KL Divergence between two multivariate Gaussian distribution
        '''

        mu = self.get_mu_params()
        mu = parameters_to_vector(mu)
        D = mu.size()[0]
        I = torch.eye(self.V.size()[1])
        if self.use_cuda:
            I = I.cuda()

        Sigma = torch.diag(F.softplus(self.diagonal)) + torch.mm(self.V, self.V.t())
        diagonal_cov_term = F.softplus(self.diagonal)

        prior_ld = torch.log(1/self.prior_prec).sum()
        Vdiagonal = torch.mm(self.V.t(), torch.diag(1/diagonal_cov_term))
        sigma_ld = torch.logdet(I + torch.mm(Vdiagonal, self.V))

        ld = prior_ld - (sigma_ld + torch.log(diagonal_cov_term).sum())
        tr = torch.trace(Sigma.mul(self.prior_prec))
        dt = torch.dot(mu.mul(self.prior_prec), mu)
        kl = tr + ld + dt - D
        return kl.div(2)


class MeanLinear(Module):
    """Applies a stochastic linear transformation to the incoming data: :math:`y = Ax + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.
    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(type(self), self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = True
            self.bias_mu = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def forward(self, input, noise_vec):
        a, b = self.weight_mu.size()
        mu_noise = noise_vec[0:a*b].view([a,b])
        perturbed_weight = self.weight_mu + mu_noise
        if self.bias is not None:
            perturbed_bias = self.bias_mu + noise_vec[a*b:]

        return F.linear(input, perturbed_weight, perturbed_bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, sigma_prior={}, bias={}'.format(
            self.in_features, self.out_features, self.sigma_prior, self.bias is not None
        )

    def reset_parameters(self):
        self.weight_mu.data.fill_(0.0)
        if self.bias is not None:
            self.bias_mu.data.fill_(0.0)
