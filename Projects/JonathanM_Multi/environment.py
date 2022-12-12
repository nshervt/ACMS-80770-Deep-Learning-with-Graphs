import numpy as np
from matplotlib import pyplot as plt

class ant_hill():
    def init(self, env_size = 16):


        ### initialize environment settings ###
        self.boxL = 16
        self.boxH = 16
        self.target_location = np.array([np.random.randint(self.boxL, size=1),np.random.randint(self.boxH , size=1)])

        #print('Target Location', (self.target_location[0],self.target_location[1]))
        ### actor start (bottom left corner) ###
        ### 4 ants (x,y, hill_king) ###
        ### hill king denotes whether ant is in the hill -> 0,1
        self.ant_locations = np.zeros((8,3))

        for i in range(len(self.ant_locations)):
            self.ant_locations[i,0] = np.random.randint(self.boxL, size=1)[0]
            self.ant_locations[i,1] = np.random.randint(self.boxH, size=1)[0]

        ### the adjacency graph per iteration ###
        self.adjacency = np.zeros((8,8))

        ### compute distance ###
        for i in range(len(self.ant_locations)):
            xi = self.ant_locations[i,0]
            yi = self.ant_locations[i,1]
            for j in range(len(self.ant_locations)):
                xj = self.ant_locations[j,0]
                yj = self.ant_locations[j,1]

                radial_distance = np.sqrt( (xi - xj)**2 + (yi - yj)**2 )

                if radial_distance < 6:
                    self.adjacency[i,j] = 1
                    self.adjacency[j,i] = 1
                else:
                    self.adjacency[i,j] = 0
                    self.adjacency[j,i] = 0

        #print("Adjacency Matrix", self.adjacency)

        self.t = 0

        self.done=False


        return self.ant_locations, self.adjacency

    def reset(self, d_message = 64):

        ### initialize environment settings ###
        self.boxL = 16
        self.boxH = 16
        self.target_location = np.array([np.random.randint(self.boxL, size=1),np.random.randint(self.boxH , size=1)])

        print('Hill Location', (self.target_location[0],self.target_location[1]))
        ### actor start (bottom left corner) ###
        ### 4 ants (x,y, hill_king) ###
        ### hill king denotes whether ant is in the hill -> 0,1
        self.ant_locations = np.zeros((8,3))

        for i in range(len(self.ant_locations)):
            self.ant_locations[i,0] = np.random.randint(self.boxL, size=1)[0]
            self.ant_locations[i,1] = np.random.randint(self.boxH, size=1)[0]

        ### the adjacency graph per iteration ###
        self.adjacency = np.zeros((8,8))

        ### compute distance ###
        for i in range(len(self.ant_locations)):
            xi = self.ant_locations[i,0]
            yi = self.ant_locations[i,1]
            for j in range(len(self.ant_locations)):
                xj = self.ant_locations[j,0]
                yj = self.ant_locations[j,1]

                radial_distance = np.sqrt( (xi - xj)**2 + (yi - yj)**2 )

                if radial_distance < 6:
                    self.adjacency[i,j] = 1
                    self.adjacency[j,i] = 1
                else:
                    self.adjacency[i,j] = 0
                    self.adjacency[j,i] = 0

        print("Adjacency Matrix", self.adjacency)

        self.t = 0

        self.done=False

        return self.ant_locations, self.adjacency


    def step(self, ant_actions):
        ### ant actions is 4x5 matrix
        ### 4 ants by 5 possible actions
        ### actions (right,left,up,down,stop)
        self.reward = 0
        ### update ant_locations ###
        for i in range(len(ant_actions)):
            if ant_actions[i,0] == 1:
                ant_loc_update = [1,0]
            elif ant_actions[i,1] == 1:
                ant_loc_update = [-1,0]
            elif ant_actions[i,2] == 1:
                ant_loc_update = [0,1]
            elif ant_actions[i,3] == 1:
                ant_loc_update = [0,-1]
            elif ant_actions[i,4] == 1:
                ant_loc_update = [0,0]

            if self.ant_locations[i,0]  + ant_loc_update[0] >= 0 and  self.ant_locations[i,0] + ant_loc_update[0] <= self.boxL:
                self.ant_locations[i,0] = self.ant_locations[i,0] + ant_loc_update[0]
            else:
                self.reward += -0.0

            if self.ant_locations[i,1] + ant_loc_update[1] >= 0 and  self.ant_locations[i,1] + ant_loc_update[1] <= self.boxH:
                self.ant_locations[i,1] = self.ant_locations[i,1] + ant_loc_update[1]

            else:
                self.reward += -0.0


        ### update ant_rewards ###
        ### each turn, each ant receives a reward which is the sum of all ants in the hill ###
        for i in range(len(self.ant_locations)):
            xi = self.ant_locations[i,0]
            yi = self.ant_locations[i,1]

            targx = self.target_location[0]
            targy = self.target_location[1]

            radial_distance = np.sqrt( (xi - targx)**2 + (yi - targy)**2 )

            if radial_distance < 2:
                self.ant_locations[i,2]=1
                self.reward += 1
            else:
                self.ant_locations[i,2]=0

        ### compute new adjacency matrix ###
        ### compute distance ###
        for i in range(len(self.ant_locations)):
            xi = self.ant_locations[i,0]
            yi = self.ant_locations[i,1]
            for j in range(len(self.ant_locations)):
                xj = self.ant_locations[j,0]
                yj = self.ant_locations[j,1]

                radial_distance = np.sqrt( (xi - xj)**2 + (yi - yj)**2 )

                if radial_distance < 8:
                    self.adjacency[i,j] = 1
                    self.adjacency[j,i] = 1
                else:
                    self.adjacency[i,j] = 0
                    self.adjacency[j,i] = 0



        self.reward += -0.1

        self.t += 1

        if self.t >= 100:
            self.done = True

        return self.ant_locations, self.reward, self.adjacency, self.done
