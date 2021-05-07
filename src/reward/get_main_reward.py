from .logp.get_reward import get_logp_score, get_penalized_logp
from .adtgpu.get_reward import get_dock_score

def get_main_reward(states, reward_type):
    if reward_type == 'logp':
        return get_logp_score(states)
    elif reward_type == 'dock':
        return get_dock_score(states)
    elif reward_type == 'plogp':
        return get_penalized_logp(states)
    else:
        raise ValueError("Reward type not recognized.")
