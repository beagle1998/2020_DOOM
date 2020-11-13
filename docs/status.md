---
layout: default
title:  Status
---


# {{ page.title }}
<iframe width="560" height="315" src="https://www.youtube.com/embed/OxZYdImZuLU" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Project Summary
	The goal of this project is to create a minecraft bot that is able to traverse a simple obstacle course, moving towards given checkpoints and a final end goal. It aims to train the bot to learn when to move and when to jump using continuous movement to avoid pits, jump over blocks and avoid deadly obstacles such as lava.

## Approach
	At the point of this report, the bot uses a DQN algorithm to map out the positive and negative values associated with certain actions. We use an action dictionary made of 7 different actions indexed 0-6 which include simple actions like moving forward, turning and jumping. The path is made of a stone block representing the start, a main path made of diamond blocks, a gold block representing the checkpoint, and an emerald block which is the final goal.
	The main obstacles used in this approach are raised blocks, which would force the bot to jump over them to continue, as well as the remainder of the world’s blocks being lava. Initially, the remainder of the world was made of grass, but for the sake of restricting the bot to the path and stopping it from meandering elsewhere, as well as to simplify learning by giving negative consequences to leaving the path, we replaced the grass blocks with lava. This allows us more control over negative consequences as well as creates a new obstacle for added complexity going forward.


## Evaluation
	Overall, the bot will be evaluated based on two main aspects: the success of reaching the destination while jumping over obstacles, and the amount of obstacles being avoided. At the point of this report, we will evaluate the bot mostly by the former goal.
	Firstly, the path will be a straight simple path with one obstacle and the final goal only, i.e. a 10-block long path with one gold block in the middle and an emerald block at the end. We will verify if the bot can successfully reach the final destination by jumping over the block in the middle of the road. When the bot passes this trivial test, we will add more blocks on the path to check if it can jump over them all.


## Remaining Goals and Challenges
	Later on, we plan to open the path more, 50 by 5 map for example, with blocks randomly put in it. The verification will then be that the bot can jump over as many gold blocks as possible while reaching the destination point. If possible, we would also like to evaluate the runtime it will take compared to us trying out the map ourselves. 
	Since we have only let the bot learn on a trivial map so far, it will be more difficult for it to learn on a more open map. We expect that the reward system will need to be modified to accommodate the change in environment. Also, we are open to changing the vision field of our bot, allowing it to see the whole map to decide the best and shortest path to jump over obstacles in the shortest time.

## Resources Used

Inspiration for the project: Minecraft Parkour Maps
https://www.minecraftmaps.com/parkour-maps

Online Example/Explanation of DQN Using PyTorch to Help Understand Its Usage:
https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda

