import torch
from actor import *
from critic import *
from memory_buffer import *

### codes the agent class ###
### responsible for choosing actions, ###
### storing actions, action exploration, ###
### and learning ###
class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
            gamma=0.8, update_actor_interval=2, warmup=1000,
            n_actions=2, max_size=100000, layer1_size=400,
            layer2_size=300, layer3_size=200, batch_size=100, noise=0.1, agent_name = 'agent1'):
        self.gamma = gamma
        self.tau = tau
        self.max_action = [1]
        self.min_action = [0]
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.agent_name = agent_name
        self.update_actor_iter = update_actor_interval
        self.noise = 0.1

        ### in TD3, there is an actor network and two critic networks ###
        ### in addition, there is a target actor network and two target critic networks ###

        ### Initialize actor network ###
        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, layer3_size, n_actions=n_actions,
                                  name='actor',agent_name = agent_name)

        ### Initializ two critic networks ###
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, layer3_size, n_actions=n_actions,
                                      name='critic_1',agent_name = agent_name)
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, layer3_size, n_actions=n_actions,
                                      name='critic_2',agent_name = agent_name)

        ### initialize target networks ###
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, layer3_size, n_actions=n_actions,
                                         name='target_actor',agent_name = agent_name)
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                         layer2_size, layer3_size, n_actions=n_actions,
                                         name='target_critic_1',agent_name = agent_name)
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                         layer2_size, layer3_size, n_actions=n_actions,
                                         name='target_critic_2',agent_name = agent_name)

        ### this sets the target networks equal to the initialized actor and critic networks ###
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, adjacency):

        ### At one point I thought this would need to be implemented here ###
        ### for some unknown reason, setting to eval mode causes poor learning ###
        ### eval mode ###
        #self.actor.eval()

        if self.agent_name == 'agent1k' or self.agent_name == 'agent2k' or self.agent_name == 'agent3k' or self.agent_name == 'agent4k':

            ### in the warmup phase, all actions are chosen randomly ###
            if self.time_step < self.warmup:
                mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)
                mu = T.abs(mu)
                mu = mu.view(8,5)

                ### agent actions ###
                Z1 = mu[0] / T.sum(mu[0])
                Z2 = mu[1] / T.sum(mu[1])
                Z3 = mu[2] / T.sum(mu[2])
                Z4 = mu[3] / T.sum(mu[3])
                Z5 = mu[4] / T.sum(mu[4])
                Z6 = mu[5] / T.sum(mu[5])
                Z7 = mu[6] / T.sum(mu[6])
                Z8 = mu[7] / T.sum(mu[7])


                ### translate to environment vector ###
                ### the environment only takes a one-hot vector ###
                ### created by taking argmax of probability vector ###

                mu1_ind = T.argmax(Z1)
                mu1_prime = Z1*0
                mu1_prime[mu1_ind] = 1

                mu2_ind = T.argmax(Z2)
                mu2_prime = Z2*0
                mu2_prime[mu2_ind] = 1

                mu3_ind = T.argmax(Z3)
                mu3_prime = Z3*0
                mu3_prime[mu3_ind] = 1

                mu4_ind = T.argmax(Z4)
                mu4_prime = Z4*0
                mu4_prime[mu4_ind] = 1

                mu5_ind = T.argmax(Z5)
                mu5_prime = Z5*0
                mu5_prime[mu5_ind] = 1

                mu6_ind = T.argmax(Z6)
                mu6_prime = Z6*0
                mu6_prime[mu6_ind] = 1

                mu7_ind = T.argmax(Z7)
                mu7_prime = Z7*0
                mu7_prime[mu7_ind] = 1

                mu8_ind = T.argmax(Z8)
                mu8_prime = Z8*0
                mu8_prime[mu8_ind] = 1


                mu1_prime = mu1_prime.cpu().detach().numpy()
                mu2_prime = mu2_prime.cpu().detach().numpy()
                mu3_prime = mu3_prime.cpu().detach().numpy()
                mu4_prime = mu4_prime.cpu().detach().numpy()
                mu5_prime = mu5_prime.cpu().detach().numpy()
                mu6_prime = mu6_prime.cpu().detach().numpy()
                mu7_prime = mu7_prime.cpu().detach().numpy()
                mu8_prime = mu8_prime.cpu().detach().numpy()

                Z1 = Z1.cpu().detach().numpy()
                Z2 = Z2.cpu().detach().numpy()
                Z3 = Z3.cpu().detach().numpy()
                Z4 = Z4.cpu().detach().numpy()

                Z5 = Z5.cpu().detach().numpy()
                Z6 = Z6.cpu().detach().numpy()
                Z7 = Z7.cpu().detach().numpy()
                Z8 = Z8.cpu().detach().numpy()

                self.time_step += 1

                ### returns the probability vectors and one-hot vectors ###
                return mu1_prime, mu2_prime, mu3_prime, mu4_prime, mu5_prime, mu6_prime, mu7_prime, mu8_prime, Z1, Z2, Z3, Z4 , Z5, Z6, Z7, Z8

            else:
                ### out of the warmup phase ###
                eps = 0.4
                state = 0


                ### the current state initial embeddings ###
                s1 = T.tensor(observation[0:60], dtype=T.float).to(self.actor.device)
                s2 = T.tensor(observation[60:120], dtype=T.float).to(self.actor.device)
                s3 = T.tensor(observation[120:180], dtype=T.float).to(self.actor.device)
                s4 = T.tensor(observation[180:240], dtype=T.float).to(self.actor.device)
                s5 = T.tensor(observation[240:300], dtype=T.float).to(self.actor.device)
                s6 = T.tensor(observation[300:360], dtype=T.float).to(self.actor.device)
                s7 = T.tensor(observation[360:420], dtype=T.float).to(self.actor.device)
                s8 = T.tensor(observation[420:480], dtype=T.float).to(self.actor.device)
                adjacency = T.tensor(observation[480:], dtype=T.float).to(self.actor.device)

                ### get the action vectors for the current state ###
                ### mu is the agent action ###
                mu = self.actor.forward(state, s1, s2, s3, s4, s5, s6, s7, s8, adjacency)
                mu = mu.view(8,5)

                ### subagent actions ###
                Z1 = mu[0]
                Z2 = mu[1]
                Z3 = mu[2]
                Z4 = mu[3]

                Z5 = mu[4]
                Z6 = mu[5]
                Z7 = mu[6]
                Z8 = mu[7]


                self.time_step += 1

                ### translate to environment vector ###
                mu1_ind = T.argmax(Z1)
                mu1_prime = Z1*0
                mu1_prime[mu1_ind] = 1

                mu2_ind = T.argmax(Z2)
                mu2_prime = Z2*0
                mu2_prime[mu2_ind] = 1

                mu3_ind = T.argmax(Z3)
                mu3_prime = Z3*0
                mu3_prime[mu3_ind] = 1

                mu4_ind = T.argmax(Z4)
                mu4_prime = Z4*0
                mu4_prime[mu4_ind] = 1

                mu5_ind = T.argmax(Z5)
                mu5_prime = Z5*0
                mu5_prime[mu5_ind] = 1

                mu6_ind = T.argmax(Z6)
                mu6_prime = Z6*0
                mu6_prime[mu6_ind] = 1

                mu7_ind = T.argmax(Z7)
                mu7_prime = Z7*0
                mu7_prime[mu7_ind] = 1

                mu8_ind = T.argmax(Z8)
                mu8_prime = Z8*0
                mu8_prime[mu8_ind] = 1



                mu1 = mu1_prime.cpu().detach().numpy()
                mu2 = mu2_prime.cpu().detach().numpy()
                mu3 = mu3_prime.cpu().detach().numpy()
                mu4 = mu4_prime.cpu().detach().numpy()

                mu5 = mu5_prime.cpu().detach().numpy()
                mu6 = mu6_prime.cpu().detach().numpy()
                mu7 = mu7_prime.cpu().detach().numpy()
                mu8 = mu8_prime.cpu().detach().numpy()



                Z1 = Z1.cpu().detach().numpy()
                Z2 = Z2.cpu().detach().numpy()
                Z3 = Z3.cpu().detach().numpy()
                Z4 = Z4.cpu().detach().numpy()
                Z5 = Z5.cpu().detach().numpy()
                Z6 = Z6.cpu().detach().numpy()
                Z7 = Z7.cpu().detach().numpy()
                Z8 = Z8.cpu().detach().numpy()

                ### Action Exploration ###
                if np.random.uniform(0,1,1)[0] < eps:

                    ### a random action turn has been triggered ###
                    if self.agent_name == 'agent1k' or self.agent_name == 'agent2k' or self.agent_name == 'agent3k' or self.agent_name == 'agent4k':
                        ### this generates 8 iid bernoullis with p = 0.5 ###
                        mu_ind = np.random.randint(2, size=8)

                        ### if the bernoulli is 1 ###
                        ### then the subagent action changes to a random action ###
                        ### else, subagent action is unchanges ###
                        if mu_ind[0] == 1:
                            mu1 = np.abs(np.random.randn(5))
                            Z1 = mu1/np.sum(mu1)
                            mu1_ind = np.argmax(mu1)
                            mu1 = mu1*0
                            mu1[mu1_ind] = 1

                        if mu_ind[1] == 1:
                            mu2 = np.abs(np.random.randn(5))
                            Z2 = mu2/np.sum(mu2)
                            mu2_ind = np.argmax(mu2)
                            mu2 = mu2*0
                            mu2[mu2_ind] = 1

                        if mu_ind[2] == 1:
                            mu3 = np.abs(np.random.randn(5))
                            Z3 = mu3/np.sum(mu3)
                            mu3_ind = np.argmax(mu3)
                            mu3 = mu3*0
                            mu3[mu3_ind] = 1

                        if mu_ind[3] == 1:
                            mu4 = np.abs(np.random.randn(5))
                            Z4 = mu4/np.sum(mu4)
                            mu4_ind = np.argmax(mu4)
                            mu4 = mu4*0
                            mu4[mu4_ind] = 1

                        if mu_ind[4] == 1:
                            mu5 = np.abs(np.random.randn(5))
                            Z5 = mu5/np.sum(mu5)
                            mu5_ind = np.argmax(mu5)
                            mu5 = mu5*0
                            mu5[mu5_ind] = 1

                        if mu_ind[5] == 1:
                            mu6 = np.abs(np.random.randn(5))
                            Z6 = mu6/np.sum(mu6)
                            mu6_ind = np.argmax(mu6)
                            mu6 = mu6*0
                            mu6[mu6_ind] = 1

                        if mu_ind[6] == 1:
                            mu7 = np.abs(np.random.randn(5))
                            Z7 = mu7/np.sum(mu7)
                            mu7_ind = np.argmax(mu7)
                            mu7 = mu7*0
                            mu7[mu7_ind] = 1

                        if mu_ind[7] == 1:
                            mu8 = np.abs(np.random.randn(5))
                            Z8 = mu8/np.sum(mu8)
                            mu8_ind = np.argmax(mu8)
                            mu8= mu8*0
                            mu8[mu8_ind] = 1



                return mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8



        self.time_step += 1

        ### train mode ###
        #self.actor.train()

        return mu_prime.cpu().detach().numpy()

    ### store tuple forfuture learning ###
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    ### learning ###
    def learn(self):

        ### only learn if the memory buffer is larger than the batch size ###
        if self.memory.mem_cntr < self.batch_size:
            return

        ### sample a batch of tuples from the buffer ###
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        if self.agent_name == 'agent1k':
            ### get subagent "next states" ###
            s1 = state_[:,0:60]
            s2 = state_[:,60:120]
            s3 = state_[:,120:180]
            s4 = state_[:,180:240]
            s5 = state_[:,240:300]
            s6 = state_[:,300:360]
            s7 = state_[:,360:420]
            s8 = state_[:,420:480]
            A = state_[:,480:]

            ### choose target actions ###
            target_actions = self.target_actor.forward(state_, s1, s2, s3, s4, s5, s6, s7, s8, A)

        ### Get current states ###
        s1p = state[:,0:60]
        s2p = state[:,60:120]
        s3p = state[:,120:180]
        s4p = state[:,180:240]
        s5p = state[:,240:300]
        s6p = state[:,300:360]
        s7p = state[:,360:420]
        s8p = state[:,420:480]
        Ap = state[:,480:]

        ### target critic values are computed using next states ###
        q1_ = self.target_critic_1.forward(state_, target_actions, s1, s2, s3, s4, s5, s6, s7, s8, A)
        q2_ = self.target_critic_2.forward(state_, target_actions, s1, s2, s3, s4, s5, s6, s7, s8, A)

        ### critic values are computed using current states ###
        q1 = self.critic_1.forward(state, action, s1p, s2p, s3p, s4p, s5p, s6p, s7p, s8p, Ap)
        q2 = self.critic_2.forward(state, action, s1p, s2p, s3p, s4p, s5p, s6p, s7p, s8p, Ap)

        ### sets all indices associated with a done flag eqaul to 0 ###
        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        ### This reduces the value estimation bias (TD3) ###
        critic_value_ = T.min(q1_, q2_)

        ### the target is computed using the target critics ###
        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1,1)

        ### learning occurs in both critic networks but not in target critic networks ###
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        ### the reward prediction error is computed using both, critics and target critics ###
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        ### actor update ###
        self.actor.optimizer.zero_grad()

        ### the actor loss is the negative of the critic value for the currently selected actions and current states ###
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state, s1p, s2p, s3p, s4p, s5p, s6p, s7p, s8p, Ap), s1p, s2p, s3p, s4p, s5p, s6p, s7p, s8p, Ap)

        ### taking the mean, but not sure if this is the best method here ###
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        ### update the target networks ###
        self.update_network_parameters()

    ### target network updates ###
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        ### get the network parameters ###
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        ### update the target networks using parameter tau ###
        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
