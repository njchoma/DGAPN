from .logp.get_reward import get_logp_score, get_penalized_logp
from .adtgpu.get_reward import get_dock_score
from .gsk3_jnk3.get_reward import get_gsk3, get_jnk3

def get_main_reward(states, reward_type, args=None):
    if reward_type == 'logp':
        return get_logp_score(states)
    elif reward_type == 'plogp':
        return get_penalized_logp(states)
    elif reward_type == 'dock':
        return get_dock_score(states, args=args)
    elif reward_type == 'gsk3':
        return get_gsk3(states)
    elif reward_type == 'jnk3':
        return get_jnk3(states)
    else:
        raise ValueError("Reward type not recognized.")
