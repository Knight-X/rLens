import numpy as np
import time

class Model:
    def __init__(self, n_iter):
        self._iter = n_iter

def pathlength(path):
    return len(path["reward"])

class PolicyGradient(Model):
    def __init__(self, n_iter, env, act, animate, min_times, max_path_length, reward_to_go):
        Model.__init__(self, n_iter)
        self._env = env
        self._act = act
        self._animate = animate
        self._min_timesteps = min_times
        self._max_pathlength = max_path_length
        self._reward_to_go = reward_to_go

    def run(self, gamma, logz, start):
        total_timesteps = 0
        for itr in range(self._iter):
            print("********** Iteration %i ************"%itr)

            # Collect paths until we have enough timesteps
            timesteps_this_batch = 0
            paths = []
            while True:
                ob, reward_map = self._env.reset()
                #ob = env.reset()
                obs, acs, rewards = [], [], []
                animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self._animate)
                steps = 0
                valid = True 
                while True:
                    if animate_this_episode:
                        env.render()
                        time.sleep(0.05)
                    obs.append(ob)
                    [distribution], acc = self._act.run(ob[None]) 
                    #[distri, acc] = act.run(ob[None])
                    #ac, valid = en.among(distri, acc[0], valid)
                    rew, ac, valid = self._env.among(distribution, reward_map, acc[0], valid)
                    acs.append(ac)
                    if valid == False:
                        steps += 1
                        rewards.append(rew)
                        continue

                    ob, done, reward_map = self._env.step(ac)
                    #ob, rew, done, _ = env.step(ac)
                    rewards.append(rew)
                    steps += 1
                    if done or steps > self._max_pathlength:
                        break
                path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs)}
                paths.append(path)
                timesteps_this_batch += pathlength(path)
                print str(timesteps_this_batch) + "go"
                if timesteps_this_batch > self._min_timesteps:
                    break
            total_timesteps += timesteps_this_batch

            # Build arrays for observation, action for the policy gradient update by concatenating 
            # across paths
            ob_no = np.concatenate([path["observation"] for path in paths])
            ac_na = np.concatenate([path["action"] for path in paths])

            # YOUR_CODE_HERE
                                                                        
                                                                        
            q_n = []
            if self._reward_to_go:
                q_n = np.concatenate([self.discount_rewards_to_go(path["reward"], gamma) for path in paths])
            else:
                q_n = np.concatenate([[self.sum_discount_rewards(path["reward"], gamma)] * pathlength(path) for path in paths])
        
            adv_n = q_n.copy()


            _, loss_value = self._act.update(ob_no, ac_na, adv_n)
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
    
    def discount_rewards_to_go(self, rewards, gamma):
        res = [] 
        future_reward = 0
        for r in reversed(rewards):
            future_reward = future_reward * gamma + r
            res.append(future_reward)
        return res[::-1]


    def sum_discount_rewards(self, rewards, gamma):
        return sum((gamma**i) * rewards[i] for i in range(len(rewards)))
