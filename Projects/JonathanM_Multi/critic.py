import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions,
            name, chkpt_dir='[filename]',agent_name='agent1'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.agent_name = agent_name
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')
        self.agent_name = agent_name

        if self.agent_name == 'agent1k' or self.agent_name == 'agent2k' or self.agent_name == 'agent3k' or self.agent_name == 'agent4k':

            ### acts as dictionary-type object with neural ntetwork layers ###
            self.p1Layers = nn.ModuleList()
            self.p2Layers = nn.ModuleList()
            self.p3Layers = nn.ModuleList()
            self.p4Layers = nn.ModuleList()
            self.p5Layers = nn.ModuleList()
            self.p6Layers = nn.ModuleList()
            self.p7Layers = nn.ModuleList()
            self.p8Layers = nn.ModuleList()

            current_dim = 3*20
            hidden_dim = [self.fc1_dims, self.fc2_dims, self.fc3_dims]
            self.ldim = len(hidden_dim)
            output_dim = 5

            ### the basic idea is that each subagent comes packaged with its own set
            ### of layers, eg. p1layers is for subagent \mu_1, etc. ###
            ### for each entry in hidden dim, a Linear -> LayerNorm -> GRU -> Linear network
            ### is generated for each subagent (8 of them)
            for hdim in hidden_dim:
                self.p1Layers.append(nn.Linear(current_dim, hdim))
                self.p1Layers.append(nn.LayerNorm(hdim))
                self.p1Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p1Layers.append(nn.Linear(current_dim*7, 7))

                self.p2Layers.append(nn.Linear(current_dim, hdim))
                self.p2Layers.append(nn.LayerNorm(hdim))
                self.p2Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p2Layers.append(nn.Linear(current_dim*7, 7))

                self.p3Layers.append(nn.Linear(current_dim, hdim))
                self.p3Layers.append(nn.LayerNorm(hdim))
                self.p3Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p3Layers.append(nn.Linear(current_dim*7, 7))

                self.p4Layers.append(nn.Linear(current_dim, hdim))
                self.p4Layers.append(nn.LayerNorm(hdim))
                self.p4Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p4Layers.append(nn.Linear(current_dim*7, 7))

                self.p5Layers.append(nn.Linear(current_dim, hdim))
                self.p5Layers.append(nn.LayerNorm(hdim))
                self.p5Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p5Layers.append(nn.Linear(current_dim*7, 7))

                self.p6Layers.append(nn.Linear(current_dim, hdim))
                self.p6Layers.append(nn.LayerNorm(hdim))
                self.p6Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p6Layers.append(nn.Linear(current_dim*7, 7))

                self.p7Layers.append(nn.Linear(current_dim, hdim))
                self.p7Layers.append(nn.LayerNorm(hdim))
                self.p7Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p7Layers.append(nn.Linear(current_dim*7, 7))

                self.p8Layers.append(nn.Linear(current_dim, hdim))
                self.p8Layers.append(nn.LayerNorm(hdim))
                self.p8Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p8Layers.append(nn.Linear(current_dim*7, 7))

                current_dim = hdim

            ### these layer constitute post-processing for applicable to the critic network only ###
            self.q1 = nn.Linear(hdim*8 + 40, 256)
            self.b1 = nn.LayerNorm(256)
            self.q2 = nn.Linear(256, 128)
            self.b2 = nn.LayerNorm(128)
            self.q3 = nn.Linear(128, 64)
            self.b3 = nn.LayerNorm(64)
            self.q4 = nn.Linear(64, 1)



        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, A):


        if self.agent_name == 'agent1k' or self.agent_name == 'agent2k' or self.agent_name == 'agent3k' or self.agent_name == 'agent4k':
            A = A.view(-1,8,8)
            ind = 0
            Z1 = Z1.view(-1,1,Z1.shape[-1])
            Z2 = Z2.view(-1,1,Z2.shape[-1])
            Z3 = Z3.view(-1,1,Z3.shape[-1])
            Z4 = Z4.view(-1,1,Z4.shape[-1])
            Z5 = Z5.view(-1,1,Z5.shape[-1])
            Z6 = Z6.view(-1,1,Z6.shape[-1])
            Z7 = Z7.view(-1,1,Z7.shape[-1])
            Z8 = Z8.view(-1,1,Z8.shape[-1])

            action = action.view(-1,1,action.shape[-1])

            ### this loops three times for the 3 message passing layers ###
            while ind < self.ldim*4:

                ### Importance Functions ###

                ### Z1 importance relations ###
                Zall1=T.cat([Z2, Z3, Z4, Z5, Z6, Z7, Z8],dim=-1)
                a1SOFT = self.p1Layers[ind+3](Zall1)
                a1SOFT = F.softmax(a1SOFT, dim=2)

                ### Z2 importance relations ###
                Zall2=T.cat([Z1, Z3, Z4, Z5, Z6, Z7, Z8],dim=-1)
                a2SOFT = self.p2Layers[ind+3](Zall2)
                a2SOFT = F.softmax(a2SOFT, dim=2)

                ### Z3 importance relations ###
                Zall3=T.cat([Z1, Z2, Z4, Z5, Z6, Z7, Z8],dim=-1)
                a3SOFT = self.p3Layers[ind+3](Zall3)
                a3SOFT = F.softmax(a3SOFT, dim=2)

                ### Z4 importance relations ###
                Zall4=T.cat([Z1, Z2, Z3, Z5, Z6, Z7, Z8],dim=-1)
                a4SOFT = self.p4Layers[ind+3](Zall4)
                a4SOFT = F.softmax(a4SOFT, dim=2)

                ### Z5 importance relations ###
                Zall5=T.cat([Z1, Z2, Z3, Z4, Z6, Z7, Z8],dim=-1)
                a5SOFT = self.p5Layers[ind+3](Zall5)
                a5SOFT = F.softmax(a5SOFT, dim=2)

                ### Z6 importance relations ###
                Zall6=T.cat([Z1, Z2, Z3, Z4, Z5, Z7, Z8],dim=-1)
                a6SOFT = self.p6Layers[ind+3](Zall6)
                a6SOFT = F.softmax(a6SOFT, dim=2)

                ### Z7 importance relations ###
                Zall7=T.cat([Z1, Z2, Z3, Z4, Z5, Z6, Z8],dim=-1)
                a7SOFT = self.p7Layers[ind+3](Zall7)
                a7SOFT = F.softmax(a7SOFT, dim=2)

                ### Z8 importance relations ###
                Zall8=T.cat([Z1, Z2, Z3, Z4, Z5, Z6, Z7],dim=-1)
                a8SOFT = self.p8Layers[ind+3](Zall8)
                a8SOFT = F.softmax(a8SOFT, dim=2)

                ### Neighbor-Importance Sums  ###
                Z1_neigh_sum = a1SOFT[:,0,0].reshape(-1,1,1)*Z2*(A[:,0,1].reshape(-1,1,1)) + a1SOFT[:,0,1].reshape(-1,1,1)*Z3*(A[:,0,2].reshape(-1,1,1)) + a1SOFT[:,0,2].reshape(-1,1,1)*Z4*(A[:,0,3].reshape(-1,1,1)) + a1SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,0].reshape(-1,1,1)) + a1SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,0].reshape(-1,1,1)) + a1SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,0].reshape(-1,1,1)) + a1SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,0].reshape(-1,1,1))
                Z2_neigh_sum = a2SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,1].reshape(-1,1,1)) + a2SOFT[:,0,1].reshape(-1,1,1)*Z3*(A[:,2,1].reshape(-1,1,1)) + a2SOFT[:,0,2].reshape(-1,1,1)*Z4*(A[:,3,1].reshape(-1,1,1)) + a2SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,1].reshape(-1,1,1)) + a2SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,1].reshape(-1,1,1)) + a2SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,1].reshape(-1,1,1)) + a2SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,1].reshape(-1,1,1))
                Z3_neigh_sum = a3SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,2].reshape(-1,1,1)) + a3SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,2].reshape(-1,1,1)) + a3SOFT[:,0,2].reshape(-1,1,1)*Z4*(A[:,3,2].reshape(-1,1,1)) + a3SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,2].reshape(-1,1,1)) + a3SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,2].reshape(-1,1,1)) + a3SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,2].reshape(-1,1,1)) + a3SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,2].reshape(-1,1,1))
                Z4_neigh_sum = a4SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,3].reshape(-1,1,1)) + a4SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,3].reshape(-1,1,1)) + a4SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,3].reshape(-1,1,1)) + a4SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,3].reshape(-1,1,1)) + a4SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,3].reshape(-1,1,1)) + a4SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,3].reshape(-1,1,1)) + a4SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,3].reshape(-1,1,1))
                Z5_neigh_sum = a5SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,4].reshape(-1,1,1)) + a5SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,4].reshape(-1,1,1)) + a5SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,4].reshape(-1,1,1)) + a5SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,4].reshape(-1,1,1)) + a5SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,4].reshape(-1,1,1)) + a5SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,4].reshape(-1,1,1)) + a5SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,4].reshape(-1,1,1))
                Z6_neigh_sum = a6SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,5].reshape(-1,1,1)) + a6SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,5].reshape(-1,1,1)) + a6SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,5].reshape(-1,1,1)) + a6SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,5].reshape(-1,1,1)) + a6SOFT[:,0,4].reshape(-1,1,1)*Z5*(A[:,4,5].reshape(-1,1,1)) + a6SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,5].reshape(-1,1,1)) + a6SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,5].reshape(-1,1,1))
                Z7_neigh_sum = a7SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,6].reshape(-1,1,1)) + a7SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,6].reshape(-1,1,1)) + a7SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,6].reshape(-1,1,1)) + a7SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,6].reshape(-1,1,1)) + a7SOFT[:,0,4].reshape(-1,1,1)*Z5*(A[:,4,6].reshape(-1,1,1)) + a7SOFT[:,0,5].reshape(-1,1,1)*Z6*(A[:,5,6].reshape(-1,1,1)) + a7SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,6].reshape(-1,1,1))
                Z8_neigh_sum = a8SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,7].reshape(-1,1,1)) + a8SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,7].reshape(-1,1,1)) + a8SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,7].reshape(-1,1,1)) + a8SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,7].reshape(-1,1,1)) + a8SOFT[:,0,4].reshape(-1,1,1)*Z5*(A[:,4,7].reshape(-1,1,1)) + a8SOFT[:,0,5].reshape(-1,1,1)*Z6*(A[:,5,7].reshape(-1,1,1)) + a8SOFT[:,0,6].reshape(-1,1,1)*Z7*(A[:,6,7].reshape(-1,1,1))

                ### Reshaping for pytorch GRU function ###
                Z1 = Z1.view(1,-1,Z1.shape[-1])
                Z2 = Z2.view(1,-1,Z2.shape[-1])
                Z3 = Z3.view(1,-1,Z3.shape[-1])
                Z4 = Z4.view(1,-1,Z4.shape[-1])
                Z5 = Z5.view(1,-1,Z5.shape[-1])
                Z6 = Z6.view(1,-1,Z6.shape[-1])
                Z7 = Z7.view(1,-1,Z7.shape[-1])
                Z8 = Z8.view(1,-1,Z8.shape[-1])

                ### GRU Aggregate Functions ###
                Z1, hn = self.p1Layers[ind+2](Z1_neigh_sum, Z1.contiguous())
                hn = hn.view(-1,1,hn.shape[-1])
                Z1 = F.relu(self.p1Layers[ind](hn))
                Z1 = self.p1Layers[ind+1](Z1)

                Z2, hn = self.p2Layers[ind+2](Z2_neigh_sum, Z2.contiguous())
                hn = hn.view(-1,1,hn.shape[-1])
                Z2 = F.relu(self.p2Layers[ind](hn))
                Z2 = self.p2Layers[ind+1](Z2)

                Z3, hn = self.p3Layers[ind+2](Z3_neigh_sum, Z3.contiguous())
                hn = hn.view(-1,1,hn.shape[-1])
                Z3 = F.relu(self.p3Layers[ind](hn))
                Z3 = self.p3Layers[ind+1](Z3)

                Z4, hn = self.p4Layers[ind+2](Z4_neigh_sum, Z4.contiguous())
                hn = hn.view(-1,1,hn.shape[-1])
                Z4 = F.relu(self.p4Layers[ind](hn))
                Z4 = self.p4Layers[ind+1](Z4)

                Z5, hn = self.p5Layers[ind+2](Z5_neigh_sum, Z5.contiguous())
                hn = hn.view(-1,1,hn.shape[-1])
                Z5 = F.relu(self.p5Layers[ind](hn))
                Z5 = self.p5Layers[ind+1](Z5)

                Z6, hn = self.p6Layers[ind+2](Z6_neigh_sum, Z6.contiguous())
                hn = hn.view(-1,1,hn.shape[-1])
                Z6 = F.relu(self.p6Layers[ind](hn))
                Z6 = self.p6Layers[ind+1](Z6)

                Z7, hn = self.p7Layers[ind+2](Z7_neigh_sum,Z7.contiguous())
                hn = hn.view(-1,1,hn.shape[-1])
                Z7 = F.relu(self.p7Layers[ind](hn))
                Z7 = self.p7Layers[ind+1](Z7)

                Z8, hn = self.p8Layers[ind+2](Z8_neigh_sum, Z8.contiguous())
                hn = hn.view(-1,1,hn.shape[-1])
                Z8 = F.relu(self.p8Layers[ind](hn))
                Z8 = self.p8Layers[ind+1](Z8)


                ind = ind + 4


            Z_out = T.cat([Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, action],dim=-1)


            ### Critic Post-Processing ###
            q1 = self.q1(Z_out)
            q1 = F.relu(q1)
            q1 = self.b1(q1)
            q1 = self.q2(q1)
            q1 = F.relu(q1)
            q1 = self.b2(q1)
            q1 = self.q3(q1)
            q1 = F.relu(q1)
            q1 = self.b3(q1)
            q1 = self.q4(q1)

            ### return approximation to the action-value funtion ###
            return q1


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file,map_location=torch.device('cpu')))
