import function as func
import numpy as np
import time
import inspect
import policy_gradient
import logz
import environment as en 

def gen(actionset):
    idx2regs = [a for a in actionset]
    idx2regs.sort()
    regs2idx = {}
    for i in range(len(idx2regs)):
        regs2idx[idx2regs[i]] = i
    return idx2regs, regs2idx

def train_PG(
            maxlength,
            idx2regs,
             regs2idx,
             height,
             weight,
             actionsize,
             exp_name,
             env_name='CartPole-v0',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000, 
             max_path_length=None,
             learning_rate=0.000000625,
             reward_to_go=True, 
             animate=True, 
             logdir=None, 
             normalize_advantages=True,
             nn_baseline=False, 
             seed=0,
             # network arguments
             n_layers=1,
             size=32,
             tofile=False,
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    np.random.seed(seed)

    # Make the gym environment
    #env = gym.make(env_name)
    env = en.Gplayer(idx2regs, regs2idx, maxlength, tofile, "./data/log/")
    act = func.ActorFunc()
    
    # Is this env continuous, or discrete?
    discrete = True
    #discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    #max_path_length = max_path_length or env.spec.max_episode_steps
    max_path_length = 3333

    # Observation and action sizes
    ob_dim = actionsize * actionsize 
    ac_dim = actionsize
    #ob_dim = env.observation_space.shape[0]
    #ac_dim = env.action_space.n if discrete else env.action_space.shape[0]


    act.createPred(ac_dim, n_layers, size)
    act.createOptimizer(learning_rate)
    act.run_init()
    #========================================================================================#
    # Training Loop
    #========================================================================================#
    pg = policy_gradient.PolicyGradient(n_iter, env, act, animate, min_timesteps_per_batch, max_path_length, reward_to_go)

    pg.run(gamma, logz, start)
