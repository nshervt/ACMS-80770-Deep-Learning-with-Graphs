import torch
from torch import nn
import numpy as np

class MyModel(nn.Module):
    def __init__(self, dim_inputlayer, dim_output):
        super(MyModel, self).__init__()
        # -- initialize layers
        self.dim_output = dim_output
        self.dim_inputlayer = dim_inputlayer
        
        torch.manual_seed(10000)
        self.conv_external   = nn.Linear(dim_inputlayer,dim_output) # GCN(5,9)
        self.conv_mu         = nn.Linear(dim_output,1)
        self.conv_log_sigma  = nn.Linear(dim_output,1)
#         self.lin2 = nn.Linear(3,3)

    def forward(self, A, h0):
        n = A.shape[-1]
        D_negative_sqrt = torch.diag_embed(1/(A+torch.eye(n)).sum(dim=2).sqrt())
        A_norm = D_negative_sqrt@(A+torch.eye(n))@D_negative_sqrt


        x = A_norm@h0
        x = self.conv_external(x)


        # output: batch_size*1
        # x = self.conv_external(A,h0)
        x = x.relu()
        
        
        x_mu = A_norm@x
        x_mu = self.conv_mu(x)
        
        x_log_sigma = A_norm@x
        x_log_sigma = self.conv_mu(x_log_sigma)
        
        
        latent_variables = x_mu  + torch.randn(self.dim_output,1) * x_log_sigma.exp()
        
        n_graphs = latent_variables.shape[0]

        latent_variables_tranpose = torch.transpose(latent_variables,1,2)
        A_hat_reconstructed = latent_variables@latent_variables_tranpose

        A_hat_reconstructed_reshape = torch.reshape(A_hat_reconstructed,(n_graphs,-1))

        A_hat_reconstructed_reshape = A_hat_reconstructed_reshape.sigmoid()

        A_hat_reconstructed = A_hat_reconstructed.sigmoid()

        return A_hat_reconstructed_reshape, A_hat_reconstructed, x_mu, x_log_sigma, latent_variables

