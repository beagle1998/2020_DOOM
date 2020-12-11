try:
    from malmo import MalmoPython
except:
    import MalmoPython

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import time
import json
import random
import time
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt 
import numpy as np
from numpy.random import randint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Hyperparameters

SIZE = 50
OBS_SIZE = 5
MAX_EPISODE_STEPS = 100
MAX_GLOBAL_STEPS = 10000
REPLAY_BUFFER_SIZE = 10000
#EPSILON_DECAY = .999
### Changing epsilon to a lower number to increase probability of agent referring to DQN
EPSILON_DECAY = .850
###
MIN_EPSILON = .1
BATCH_SIZE = 128
GAMMA = .9
TARGET_UPDATE = 100
LEARNING_RATE = 1e-4
START_TRAINING = 500
LEARN_FREQUENCY = 1

##### Adding boolean values to check if a continuous action is already being done ###
##
##isMoving = False
##isJumping = False
##isTurningLeft = False
##isTurningRight = False
###isNotTurning = True
##
#####

ACTION_DICT = {
    0: 'move 1', # Move forward at normal speed
    1: 'move 0', # Stop moving
    2: 'turn -0.5',# Turn to the left
    3: 'turn 0', # Stop turning
    4: 'turn 0.5', # turn to the right
    5: 'jump 1', # start jumping
    6: 'jump 0'  # stop jumping
}

reward1=0

# Q-Value Network
class QNetwork(nn.Module):
    #------------------------------------
    #
    #   TODO: Modify network architecture
    #
    #-------------------------------------

    def __init__(self, obs_size, action_size, hidden_size=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(np.prod(obs_size), hidden_size),
                                 nn.LeakyReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.LeakyReLU(),
                                 nn.Linear(hidden_size, action_size))
        
    def forward(self, obs):
        """
        Estimate q-values given obs

        Args:
            obs (tensor): current obs, size (batch x obs_size)

        Returns:
            q-values (tensor): estimated q-values, size (batch x action_size)
        """
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)
        return self.net(obs_flat)

# quit when reach parkour end or max steps reached


# diamond block: path, gold block: checkpoint, emerald block: mission end
def drawPath(pathID):
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


### Will be changing lava reward to be less harsh (-20 -> -5). Hopefully this will stop the agent
### from jumping and turning in circles since there will be less of a negative reward moving forward
def GetMissionXML():

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
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='lava'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            drawPath(3) + \
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
                                <min x="-'''+str(int(OBS_SIZE/2))+'''" y="-1" z="-'''+str(int(OBS_SIZE/2))+'''"/>
                                <max x="'''+str(int(OBS_SIZE/2))+'''" y="0" z="'''+str(int(OBS_SIZE/2))+'''"/>
                            </Grid>
                        </ObservationFromGrid>
                        <RewardForTouchingBlockType>
                            <Block reward="30" type="gold_block"/>
                            <Block reward="50" type="emerald_block"/>
                            <Block reward="10" type="diamond_block"/>
                            <Block reward="-5" type="lava"/>
                            <Block reward="-10" type="stone"/>
                        </RewardForTouchingBlockType>
                        <RewardForTimeTaken initialReward="1000" delta="-1" density="PER_TICK"/>
                        <AgentQuitFromTouchingBlockType>
                            <Block type ="emerald_block"/>
                        </AgentQuitFromTouchingBlockType>
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''


def get_action(obs, q_network, epsilon, allow_break_action):
    """
    Select action according to e-greedy policy

    Args:
        obs (np-array): current observation, size (obs_size)
        q_network (QNetwork): Q-Network
        epsilon (float): probability of choosing a random action

    Returns:
        action (int): chosen action [0, action_size)
    """
    #   TODO: Implement e-greedy policy

##    global isMoving
##    global isJumping
##    global isTurningLeft
##    global isTurningRight
    
    if random.random() < epsilon:
        return randint(0,7)
        #print(f"Random: {action}")
        ### Expanding to ensure random choice doesn't pick reduntant actions, ensuring the agent is always picking a new action (for example, not choosing to jump over and over when it is already jumping)
        ### Also keeps track of what actions are being taken. For example, if the agent started jumping, it should track that it is currently jumping
##        while True:
##            action = randint(0,7)
##            if (isMoving and action == 0) or (isJumping and action == 5) or (isTurningLeft and action == 2) or (isTurningRight and action == 4):
##                continue
##            elif ((not isMoving) and action == 1) or ((not isJumping) and action == 6) or (not (isTurningLeft) and (not isTurningRight) and action == 3):
##                continue
##            else:
##                ### Added print call to be able to see the exact commands chosen when random choice is picked###
##                print(f"Random: {action}")
##                if action == 0:
##                    isMoving = True
##                elif action == 1:
##                    isMoving = False
##                elif action == 2:
##                    isTurningRight = False
##                    isTurningLeft = True
##                elif action == 3:
##                    isTurningLeft = False
##                    isTurningRight = False
##                elif action == 4:
##                    isTurningLeft = False
##                    isTurningRight = True
##                elif action == 5:
##                    isJumping = True
##                elif action == 6:
##                    isJumping = False
##                return action

    with torch.no_grad():
        obs_torch = torch.tensor(obs.copy(), dtype = torch.float).unsqueeze(0)
        action_values = q_network(obs_torch)

        action_idx = torch.argmax(action_values).item()
        
    #print(f"Chosen: {action_idx}")
##    if action_idx == 0:
##        isMoving = True
##    elif action_idx == 1:
##        isMoving = False
##    elif action_idx == 2:
##        isTurningLeft = True
##    elif action_idx == 3:
##        isTurningLeft = False
##        isTurningRight = False
##    elif action_idx == 4:
##        isTurningRight = True
##    elif action_idx == 5:
##        isJumping = True
##    elif action_idx == 6:
##        isJumping = False
        
    return action_idx


def init_malmo(agent_host):
    """
    Initialize new malmo mission.
    """
    my_mission = MalmoPython.MissionSpec(GetMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)

    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_clients, my_mission_record, 0, "Parkour_Bot" )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    return agent_host


def get_observation(world_state):
    """
    Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
    The agent is in the center square facing up.

    Args
        world_state: <object> current agent world state

    Returns
        observation: <np.array>
    """
    obs = np.zeros((2, OBS_SIZE, OBS_SIZE))

    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            # First we get the json from the observation API
            msg = world_state.observations[-1].text
            observations = json.loads(msg)

            # Get observation
            grid = observations['floorAll']
            grid_binary = [1 if x == 'diamond_ore' or x == 'lava' else 0 for x in grid]
            obs = np.reshape(grid_binary, (2, OBS_SIZE, OBS_SIZE))

            # Rotate observation with orientation of agent
            yaw = observations['Yaw']
            if yaw == 270:
                obs = np.rot90(obs, k=1, axes=(1, 2))
            elif yaw == 0:
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif yaw == 90:
                obs = np.rot90(obs, k=3, axes=(1, 2))
            
            break

    return obs


def prepare_batch(replay_buffer):
    """
    Randomly sample batch from replay buffer and prepare tensors

    Args:
        replay_buffer (list): obs, action, next_obs, reward, done tuples

    Returns:
        obs (tensor): float tensor of size (BATCH_SIZE x obs_size
        action (tensor): long tensor of size (BATCH_SIZE)
        next_obs (tensor): float tensor of size (BATCH_SIZE x obs_size)
        reward (tensor): float tensor of size (BATCH_SIZE)
        done (tensor): float tensor of size (BATCH_SIZE)
    """
    batch_data = random.sample(replay_buffer, BATCH_SIZE)
    obs = torch.tensor([x[0] for x in batch_data], dtype=torch.float)
    action = torch.tensor([x[1] for x in batch_data], dtype=torch.long)
    next_obs = torch.tensor([x[2] for x in batch_data], dtype=torch.float)
    reward = torch.tensor([x[3] for x in batch_data], dtype=torch.float)
    done = torch.tensor([x[4] for x in batch_data], dtype=torch.float)
    
    return obs, action, next_obs, reward, done
  

def learn(batch, optim, q_network, target_network):
    """
    Update Q-Network according to DQN Loss function

    Args:
        batch (tuple): tuple of obs, action, next_obs, reward, and done tensors
        optim (Adam): Q-Network optimizer
        q_network (QNetwork): Q-Network
        target_network (QNetwork): Target Q-Network
    """
    obs, action, next_obs, reward, done = batch

    optim.zero_grad()
    values = q_network(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
    target = torch.max(target_network(next_obs), 1)[0]
    target = reward + GAMMA * target * (1 - done)
    loss = torch.mean((target - values) ** 2)
    loss.backward()
    optim.step()

    return loss.item()


def log_returns(steps, returns):
    """
    Log the current returns as a graph and text file

    Args:
        steps (list): list of global steps after each episode
        returns (list): list of total return of each episode
    """
    box = np.ones(10) / 10
    returns_smooth = np.convolve(returns, box, mode='same')
    plt.clf()
    plt.plot(steps, returns_smooth)
    plt.title('Parkour_Bot')
    plt.ylabel('Return')
    plt.xlabel('Steps')
    plt.savefig('returns.png')

    with open('returns.txt', 'w') as f:
        for value in returns:
            f.write("{}\n".format(value)) 


def train(agent_host):
    """
    Main loop for the DQN learning algorithm

    Args:
        agent_host (MalmoPython.AgentHost)
    """
    # Init networks
    q_network = QNetwork((2, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network = QNetwork((2, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    # Init replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Init vars
    global_step = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    steps = []

    # Begin main loop
    loop = tqdm(total=MAX_GLOBAL_STEPS, position=0, leave=False)
    while global_step < MAX_GLOBAL_STEPS:
        episode_step = 0
        episode_return = 0
        episode_loss = 0
        done = False

        # Setup Malmo
        agent_host = init_malmo(agent_host)
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:",error.text)
        obs = get_observation(world_state)

        # Run episode
        while world_state.is_mission_running:
            # Get action
            allow_break_action = obs[1, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 1
            action_idx = get_action(obs, q_network, epsilon, allow_break_action)
            command = ACTION_DICT[action_idx]

            # Take step
            agent_host.sendCommand(command)

            # If your agent isn't registering reward you may need to increase this
            time.sleep(0.2)

            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            if episode_step >= MAX_EPISODE_STEPS or \
                    (obs[0, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 1 and \
                    obs[1, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 0 and \
                    command == 'move 1'):
                done = True
                time.sleep(2)  

            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            next_obs = get_observation(world_state) 

            # Get reward
            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
            episode_return += reward

            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs

            # Learn
            global_step += 1
            if global_step > START_TRAINING and global_step % LEARN_FREQUENCY == 0:
                batch = prepare_batch(replay_buffer)
                loss = learn(batch, optim, q_network, target_network)
                episode_loss += loss

                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY

                if global_step % TARGET_UPDATE == 0:
                    target_network.load_state_dict(q_network.state_dict())

        num_episode += 1
        returns.append(episode_return)
        steps.append(global_step)
        avg_return = sum(returns[-min(len(returns), 10):]) / min(len(returns), 10)
        loop.update(episode_step)
        loop.set_description('Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.2f}'.format(
            num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        if num_episode > 0 and num_episode % 10 == 0:
            log_returns(steps, returns)
            print()


if __name__ == '__main__':
    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    train(agent_host)

















