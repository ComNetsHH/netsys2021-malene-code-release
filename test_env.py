#   Code to replicate results of the publication 
#   'Time- and Frequency-Domain Dynamic Spectrum Access: Learning Cyclic Medium Access Patterns in Partially Observable Environments' 
#   published at the Conference on Networked Systems (NetSys) 2021, Lübeck, Germany.
#   https://github.com/ComNetsHH/netsys2021-malene-code-release
#
#     Copyright (C) 2021  Institute of Communication Networks, 
# 			  	          Hamburg University of Technology,
# 			  	          https://www.tuhh.de/comnets
# 			  	(C) 2021  Sebastian Lindner, 
# 			  	          sebastian.lindner@tuhh.de
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.


import unittest
from env import *
import matplotlib.pyplot as plt
import numpy as np
import os


class TestDMEEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = DMEEnvironment(5)
        self.env.add(request_channel=0, response_channel=1, periodicity=4, offset=0)
        self.env.add(request_channel=0, response_channel=1, periodicity=4, offset=3)
        self.env.add(request_channel=3, response_channel=4, periodicity=3, offset=1)

    def test_steps(self):
        t_max = 30
        M = np.zeros((t_max, self.env.n_channels))
        print('Fully observable observations:')
        for t in range(t_max):
            state = self.env.step()
            M[t] = state
            self.assertTrue(i == DMEEnvironment.IDLE_CHANNEL_STATE or i == DMEEnvironment.BUSY_CHANNEL_STATE for i in state)
            print("t=" + str(t+1) + ": " + str(state))
        print()
            
        ## Uncomment to show a resource utilization plot        
        # plt.rcParams.update({'font.size': 9})
        # plt.imshow(np.transpose(-1*M), cmap='Greys', origin='lower')        
        # plt.xlabel('time slot t')
        # plt.ylabel('channel index')
        # ax = plt.gca()
        # ax.set_yticks(range(self.env.n_channels))
        # plt.show()


class TestDMEEnvironmentPartiallyObservable(unittest.TestCase):
    def setUp(self):
        self.env = DMEEnvironmentPartiallyObservable(5)
        self.env.add(request_channel=0, response_channel=1, periodicity=4, offset=0)
        self.env.add(request_channel=0, response_channel=1, periodicity=4, offset=3)
        self.env.add(request_channel=3, response_channel=4, periodicity=3, offset=1)

    def test_steps(self):
        t_max = 30
        M = np.zeros((t_max, self.env.n_channels))
        print('Partially observable observations:')
        for t in range(t_max):
            state = self.env.step(action=np.random.randint(0, self.env.n_channels))
            M[t] = state
            self.assertTrue(i == DMEEnvironment.IDLE_CHANNEL_STATE or i == DMEEnvironment.BUSY_CHANNEL_STATE or i == DMEEnvironmentPartiallyObservable.UNKNOWN_CHANNEL_STATE for i in state)
            self.assertTrue(np.sum(state) == 1 or np.sum(state) == -1)
            print("t=" + str(t+1) + ": " + str(state))        
        print()
        
        ## Uncomment to show a resource utilization plot        
        # plt.rcParams.update({'font.size': 9})
        # plt.imshow(np.transpose(-1*M), cmap='Greys', origin='lower')
        # cbar = plt.colorbar()
        # cbar.set_ticks([-1, 0, 1])
        # cbar.set_ticklabels(['idle', 'no info', 'busy'])
        # plt.xlabel('time slot t')
        # plt.ylabel('channel index')
        # ax = plt.gca()
        # ax.set_yticks(range(self.env.n_channels))
        # plt.show()


if __name__ == '__main__':
    unittest.main()
