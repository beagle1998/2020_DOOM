def drawStraightPath():
    path = ""
    for i in range(10):
        path += f"<DrawBlock x='0' y='1' z='{i}' type='diamond_block' />" \
                f"<DrawBlock x='1' y='1' z='{i}' type='stone' />" \
                f"<DrawBlock x='-1' y='1' z='{i}' type='stone' />"

    path += "<DrawBlock x='0' y='1' z='5' type='gold_block' />"
    path += "<DrawBlock x='0' y='1' z='10' type='emerald_block' />"
    return path


def drawPath2():
    path = ""
    for i in range(10):
        path += f"<DrawBlock x='0' y='1' z='{i}' type='diamond_block' />" \
                f"<DrawBlock x='1' y='1' z='{i}' type='stone' />" \
                f"<DrawBlock x='-1' y='1' z='{i}' type='stone' />"

    path += "<DrawBlock x='0' y='2' z='5' type='gold_block' />"
    path += "<DrawBlock x='0' y='2' z='10' type='emerald_block' />"
    return path

def drawPath3():
    path = ""
    for i in range(5):
        path += f"<DrawBlock x='0' y='1' z='{i}' type='diamond_block' />" \
                f"<DrawBlock x='1' y='1' z='{i}' type='stone' />" \
                f"<DrawBlock x='-1' y='1' z='{i}' type='stone' />"
    
    for i in range(6):
        path += f"<DrawBlock x='{i}' y='1' z='5' type='diamond_block' />" \
                f"<DrawBlock x='{i+1}' y='1' z='4' type='stone' />" \
                f"<DrawBlock x='{i+1}' y='1' z='6' type='stone' />"
    path += "<DrawBlock x='-1' y='1' z='5' type='stone' /><DrawBlock x='-1' y='1' z='6' type='stone' /><DrawBlock x='0' y='1' z='6' type='stone' />"

    path += "<DrawBlock x='0' y='1' z='5' type='gold_block' />"
    path += "<DrawBlock x='6' y='1' z='5' type='emerald_block' />"
    return path


def drawPath4():
    path = ""
    for i in range(7):
        path += f"<DrawBlock x='0' y='{i+1}' z='{i}' type='diamond_block' />" \
                f"<DrawBlock x='1' y='1' z='{i}' type='stone' />" \
                f"<DrawBlock x='-1' y='1' z='{i}' type='stone' />"

    path += "<DrawBlock x='0' y='4' z='3' type='gold_block' />"
    path += "<DrawBlock x='0' y='7' z='7' type='emerald_block' />"
    return path