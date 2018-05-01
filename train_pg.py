import numpy as np
import logz
import tensorflow as tf
import environment as en 
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process
import gym
import function as func 
import policy_gradient

import socket
HOST = '127.0.0.1'
PORT = 1992
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))





#============================================================================================#
# Policy Gradient
#============================================================================================#

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
    env = en.Gplayer(sock, idx2regs, regs2idx, maxlength, tofile)
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

def gen(actionset):
    idx2regs = [a for a in actionset]
    idx2regs.sort()
    regs2idx = {}
    for i in range(len(idx2regs)):
        regs2idx[idx2regs[i]] = i
    return idx2regs, regs2idx

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.00000625)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=128)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None
    rplayer = en.RandomPlayer(sock)
    actionset, maxlength = rplayer.interact()
    idx2regs, regs2idx = gen(actionset)
    name = "./data/log/register maping.txt"
    f = open(name, "w")
    f.write(str(regs2idx))
    f.close()

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                maxlength,
                idx2regs,
                regs2idx,
                247,
                247,
                len(idx2regs),
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate, 
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                tofile=False
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        #p = Process(target=train_func, args=tuple())
        #p.start()
        #p.join()
        train_func()
        

if __name__ == "__main__":
    main()
