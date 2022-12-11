import numpy as np
import torch
from actor import *
from critic import *
from memory_buffer import *
from agent import *
from environment import *

if __name__ == '__main__':

    agent1k = Agent(alpha=0.00001, beta=0.00001,
                input_dims=8*3*20+8*8, tau=0.001,
                env=0, batch_size=32, layer1_size=256, layer2_size=128, layer3_size=64,
                n_actions=40, agent_name = 'agent1k')

    ### number of games ###
    n_games = 2500

    ### keep track of scores for averaging ###
    score_history = []

    ### load pre-existing models ###
    agent1k.load_models()

    ### creating observations for each subagent ###
    observation1 = np.zeros(3*20)
    observation2 = np.zeros(3*20)
    observation3 = np.zeros(3*20)
    observation4 = np.zeros(3*20)
    observation5 = np.zeros(3*20)
    observation6 = np.zeros(3*20)
    observation7 = np.zeros(3*20)
    observation8 = np.zeros(3*20)


    env = ant_hill()
    env.init()
    ant_locations, adjacency = env.reset()

    for i in range(n_games):
        ### initialize and get ant locations ###
        ### each ant agent take its location and hill_staus as observation
        ant_locations, adjacency = env.reset()


        ### new observation1k ###
        ### storing initial embedding ###
        observation1k = np.zeros(8*3*20 + 8*8)
        observation1k[0:57] = observation1k[3:60]
        observation1k[57:60] = ant_locations[0]
        observation1k[60:117] = observation1k[63:120]
        observation1k[117:120] = ant_locations[1]
        observation1k[120:177] = observation1k[123:180]
        observation1k[177:180] = ant_locations[2]
        observation1k[180:237] = observation1k[183:240]
        observation1k[237:240] = ant_locations[3]
        observation1k[240:297] = observation1k[243:300]
        observation1k[297:300] = ant_locations[4]
        observation1k[300:357] = observation1k[303:360]
        observation1k[357:360] = ant_locations[5]
        observation1k[360:417] = observation1k[363:420]
        observation1k[417:420] = ant_locations[6]
        observation1k[420:477] = observation1k[423:480]
        observation1k[477:480] = ant_locations[7]
        observation1k[480:544] = adjacency.reshape(8*8)



        done = False
        score = 0
        turn = 0
        while not done:


            ### Put actions together ###
            action_env = np.zeros((8,5))



            ### agent1k action ###
            ### get raw probability vectors ###
            action1k_raw = agent1k.choose_action(observation1k,adjacency)



            ### construct action ###
            ### this is environment action, so one-hot ###
            action_env = np.zeros((8,5))
            action_env[0,:] = np.round(action1k_raw[0])
            action_env[1,:] = np.round(action1k_raw[1])
            action_env[2,:] = np.round(action1k_raw[2])
            action_env[3,:] = np.round(action1k_raw[3])
            action_env[4,:] = np.round(action1k_raw[4])
            action_env[5,:] = np.round(action1k_raw[5])
            action_env[6,:] = np.round(action1k_raw[6])
            action_env[7,:] = np.round(action1k_raw[7])


            ### get new subagent locations ###
            ant_locations, reward, adjacency, done = env.step(action_env)


            ### set new initial embeddings ###
            observation1_ = ant_locations[0]
            observation2_ = ant_locations[1]
            observation3_ = ant_locations[2]
            observation4_ = ant_locations[3]
            observation5_ = ant_locations[4]
            observation6_ = ant_locations[5]
            observation7_ = ant_locations[6]
            observation8_ = ant_locations[7]



            ### new observation1k ###
            ### store as really long vector ###
            observation1k_ = np.zeros(8*3*20 + 8*8)
            observation1k_[0:57] = observation1k_[3:60]
            observation1k_[57:60] = ant_locations[0]
            observation1k_[60:117] = observation1k_[63:120]
            observation1k_[117:120] = ant_locations[1]
            observation1k_[120:177] = observation1k_[123:180]
            observation1k_[177:180] = ant_locations[2]
            observation1k_[180:237] = observation1k_[183:240]
            observation1k_[237:240] = ant_locations[3]
            observation1k_[240:297] = observation1k_[243:300]
            observation1k_[297:300] = ant_locations[4]
            observation1k_[300:357] = observation1k_[303:360]
            observation1k_[357:360] = ant_locations[5]
            observation1k_[360:417] = observation1k_[363:420]
            observation1k_[417:420] = ant_locations[6]
            observation1k_[420:477] = observation1k_[423:480]
            observation1k_[477:480] = ant_locations[7]
            observation1k_[480:544] = adjacency.reshape(8*8)

            action1k = np.zeros(8*5)
            action1k = action_env.reshape(8*5)

            ### for learning, the probability vectors are used ###
            ### could be an issue here, maybe it should be the one-hots? ###
            action_learn = np.zeros((8,5))
            action_learn[0,:] = action1k_raw[8]
            action_learn[1,:] = action1k_raw[9]
            action_learn[2,:] = action1k_raw[10]
            action_learn[3,:] = action1k_raw[11]
            action_learn[4,:] = action1k_raw[12]
            action_learn[5,:] = action1k_raw[13]
            action_learn[6,:] = action1k_raw[14]
            action_learn[7,:] = action1k_raw[15]

            ### remember() takes in vector argument for actions, not matrix ###
            action_learn = action_learn.reshape(8*5)

            ### currently learning is commented out ###
            ### uncomment to start learning ###
            ### agent1k learn ###
            # agent1k.remember(observation1k, action_learn, reward, observation1k_, done)
            # agent1k.learn()

            ### set new observation  as current ###
            observation1k = observation1k_

            score += reward
            turn += 1

        if turn%100 == 0:
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            print('action_env',action_env)

            if i%10 == 0:
                agent1k.save_models()
            print('episode ', i, 'score %.2f' % score,
                    'trailing 100 games avg %.3f' % avg_score)

#agent1k.save_models()
