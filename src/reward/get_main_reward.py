from .logp.get_reward import get_logp_score
from .adtgpu.get_reward import get_dock_score

def get_main_reward(states, reward_type):
    if reward_type == 'logp':
        return get_logp_score(states)
    elif reward_type == 'dock':
        return get_dock_score(states)
    else:
        raise ValueError("Reward type not recognized.")
