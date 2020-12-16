import random

def drawStraightPath():
    path = ""
    for i in range(10):
        path += f"<DrawBlock x='0' y='1' z='{i}' type='diamond_block' />"

    path += "<DrawBlock x='0' y='1' z='5' type='gold_block' />"
    path += "<DrawBlock x='0' y='1' z='10' type='emerald_block' />"
    return path


def drawPath2():
    path = ""
    for i in range(10):
        path += f"<DrawBlock x='0' y='1' z='{i}' type='diamond_block' />"

    path += "<DrawBlock x='0' y='2' z='5' type='gold_block' />"
    path += "<DrawBlock x='0' y='2' z='10' type='emerald_block' />"
    return path

def drawPath3():
    path = ""
    for i in range(5):
        path += f"<DrawBlock x='0' y='1' z='{i}' type='diamond_block' />"
    
    for i in range(6):
        path += f"<DrawBlock x='{i}' y='1' z='5' type='diamond_block' />"

    path += "<DrawBlock x='0' y='1' z='5' type='gold_block' />"
    path += "<DrawBlock x='6' y='1' z='5' type='emerald_block' />"
    return path


def drawPath4():
    path = ""
    for i in range(7):
        path += f"<DrawBlock x='0' y='{i+1}' z='{i}' type='diamond_block' />" \

    path += "<DrawBlock x='0' y='4' z='3' type='gold_block' />"
    path += "<DrawBlock x='0' y='7' z='7' type='emerald_block' />"
    return path


rand_noise = 0.2
def sixBsix():
    path = ""
    for i in range(7):
        for j in range(7):
            path += f"<DrawBlock x='{i}' y='0' z='{j}' type='stone' />"
            if(random.random() < rand_noise):
                path += f"<DrawBlock x='{i}' y='1' z='{j}' type='stone' />"
    path += f"<DrawBlock x='6' y='0' z='6' type='emerald_block' /><DrawBlock x='6' y='1' z='6' type='air' />"
