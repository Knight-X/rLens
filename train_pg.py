import numpy as np
import tensorflow as tf
import environment as en 
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process
import gym
import function as func 

import socket
HOST = '127.0.0.1'
PORT = 1992
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))


def pathlength(path):
    return len(path["reward"])



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

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob, reward_map = env.reset()
            #ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            valid = True 
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                [distri], acc = act.run(ob[None]) 
                #[distri, acc] = act.run(ob[None])
                #ac, valid = en.among(distri, acc[0], valid)
                rew, ac, valid = env.among(distri, reward_map, acc[0], valid)
                acs.append(ac)
                if valid == False:
                  steps += 1
                  #rew = -0.01
                  rewards.append(rew)
                  continue

                ob, done, reward_map = env.step(ac)
                #ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            print str(timesteps_this_batch) + "go"
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])

        # YOUR_CODE_HERE
        def discount_rewards_to_go(rewards, gamma):
            res = [] 
            future_reward = 0
            for r in reversed(rewards):
                future_reward = future_reward * gamma + r
                res.append(future_reward)
            return res[::-1]
                                                                        
                                                                        
        def sum_discount_rewards(rewards, gamma):
            return sum((gamma**i) * rewards[i] for i in range(len(rewards)))
        q_n = []
        if reward_to_go:
            q_n = np.concatenate([discount_rewards_to_go(path["reward"], gamma) for path in paths])
        else:
            q_n = np.concatenate([[sum_discount_rewards(path["reward"], gamma)] * pathlength(path) for path in paths])

        adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1. 
            # YOUR_CODE_HERE
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
            adv_n = adv_n * (1.0 + 1e-8) + 10.0


        # Log diagnostics
        _, loss_value = act.update(ob_no, ac_na, adv_n)
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("Loss", loss_value)
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()

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
