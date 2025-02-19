


import random

def bad_manager_node(state):
    node_num = random.randint(1, 2)
    return dict(
        input = state['input'],
        assigned_node = node_num,
        prev_node_to_write = 'manager'
    )


def node_1(state):
    return dict(
        input = state['input'],
        assigned_node = 1 if 'node 1' in state['input'].content else 2,
        pred_node_to_write = 'node 1'

    )

def node_2(state):
    return dict(
        input = state['input'],
        assigned_node = 2 if 'node 1' in state['input'].content else 1,
        prev_node_to_write = 'node 2'

    )


