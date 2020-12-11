try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

import gym, ray
from gym.spaces import Box
from ray.rllib.agents import ppo

class ParkourBot(gym.Env):

    def __init__(self, env_config):
        # Static Parameters
        self.size = 50
        self.reward_density = .1
        self.penalty_density = .02
        self.obs_size = 5
        self.max_episode_steps = 100
        self.log_frequency = 10

        # Rllib Parameters
        self.action_space = Box(-1, 1, shape=(3,), dtype=np.float32)
        self.observation_space = Box(0, 1, shape=(np.prod([2, self.obs_size, self.obs_size]), ), dtype=np.int32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # ParkourBot Parameters
        self.obs = None
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        # Log
        if len(self.returns) > self.log_frequency and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs.flatten()

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        # Get Action
        if action[2] > 0:
            self.agent_host.sendCommand('jump 1')
        else:
            self.agent_host.sendCommand('jump 0')
        self.agent_host.sendCommand('move {:30.1f}'.format(action[0]))
        self.agent_host.sendCommand('turn {:30.1f}'.format(action[1]))
        
        time.sleep(.2)
        
        self.episode_step += 1

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state) 

        # Get Done
        done = not world_state.is_mission_running 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward

        return self.obs.flatten(), reward, done, dict()

    def drawPath(self, pathID):
        '''
        This function will draw the path specified by pathID. For personal preference,
        I am keeping these paths in the same file for ease of editing for testing purposes
        '''
        path = ''
        if pathID == 0:
            for i in range(10):
                path += f"<DrawBlock x='0'  y='1' z='{i}' type='diamond_block' />"
            path += "<DrawBlock x='0'  y='2' z='5' type='gold_block' />"
            path += "<DrawBlock x='0'  y='1' z='10' type='emerald_block' />"

        elif pathID == 1:
            for i in range(10):
                path += f"<DrawBlock x='0' y='1' z='{i}' type='diamond_block' />" \

            path += "<DrawBlock x='0' y='2' z='5' type='gold_block' />"
            path += "<DrawBlock x='0' y='2' z='10' type='emerald_block' />"

        elif pathID == 2:
            for i in range(5):
                path += f"<DrawBlock x='0' y='1' z='{i}' type='diamond_block' />" \
        
            for i in range(6):
                path += f"<DrawBlock x='{i}' y='1' z='5' type='diamond_block' />" \

            path += "<DrawBlock x='0' y='1' z='5' type='gold_block' />"
            path += "<DrawBlock x='6' y='1' z='5' type='emerald_block' />"

        elif pathID == 3:
            for i in range(7):
                path += f"<DrawBlock x='0' y='{i+1}' z='{i}' type='diamond_block' />" \

            path += "<DrawBlock x='0' y='4' z='3' type='gold_block' />"
            path += "<DrawBlock x='0' y='7' z='7' type='emerald_block' />"

        return path

    def GetMissionXML(self):
        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>Parkour_Bot</Summary>
                    </About>

                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>12000</StartTime>
                                <AllowPassageOfTime>true</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <DrawingDecorator>''' + \
                                "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-self.size, self.size, -self.size, self.size) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='lava'/>".format(-self.size, self.size, -self.size, self.size) + \
                                self.drawPath(2) + \
                                '''<DrawBlock x='0'  y='2' z='0' type='air' />
                                <DrawBlock x='0'  y='1' z='0' type='stone' />
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>ParkourBot</Name>
                        <AgentStart>
                            <Placement x="0.5" y="2" z="0.5" pitch="45" yaw="0"/>
                        </AgentStart>
                        <AgentHandlers>
                            <ContinuousMovementCommands/>
                            <ObservationFromFullStats/>
                            <ObservationFromGrid>
                                <Grid name="floorAll">
                                    <min x="-'''+str(int(self.obs_size/2))+'''" y="-1" z="-'''+str(int(self.obs_size/2))+'''"/>
                                    <max x="'''+str(int(self.obs_size/2))+'''" y="0" z="'''+str(int(self.obs_size/2))+'''"/>
                                </Grid>
                            </ObservationFromGrid>
                            <RewardForTouchingBlockType>
                                <Block reward="50" type="gold_block" behaviour="oncePerBlock"/>
                                <Block reward="100" type="emerald_block"/>
                                <Block reward="10" type="diamond_block" behaviour="oncePerBlock"/>
                                <Block reward="-1" type="lava" behaviour="oncePerBlock"/>
                                <Block reward="-1" type="stone" behaviour="oncePerBlock"/>
                            </RewardForTouchingBlockType>
                            <RewardForMissionEnd rewardForDeath="-1">
                                <Reward reward="0" description="Mission End"/>
                            </RewardForMissionEnd>
                            <RewardForTimeTaken initialReward="1000" delta="-1" density="PER_TICK"/>
                            <AgentQuitFromTouchingBlockType>
                                <Block type ="emerald_block"/>
                            </AgentQuitFromTouchingBlockType>
                            <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''




    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.GetMissionXML(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'ParkourBot' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state



    def get_observation(self, world_state):
        obs = np.zeros((2, self.obs_size, self.obs_size))

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)

                # Get Observation
                grid = observations['floorAll']
                grid_binary = [1 if x == 'diamond_block' or x == 'gold_block' or x == 'emerald_block' or x == 'lava' else 0 for x in grid]

                obs = np.reshape(grid_binary, (2, self.obs_size, self.obs_size))

                yaw = observations['Yaw']
                if yaw == 270:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw == 0:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw == 90:
                    obs = np.rot90(obs, k=3, axes=(1, 2))

                break
        return obs

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns, box, mode='same')
        plt.clf()
        plt.plot(self.steps, returns_smooth)
        plt.title('ParkoutBot')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps, self.returns):
                f.write("{}\t{}\n".format(step, value))


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=ParkourBot, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
                    


































                

        
