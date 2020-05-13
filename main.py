"""
@ Author: Pky
@ Time: 2020/2/2
@ Software: PyCharm 
"""
import json
from RedTeam.RedTeam import RedTeam
from BlueTeam.BlueTeam import BlueTeam
from EMCSimulator import EMCSimulator
from AlgorithmsLib.PolicyGradient import PolicyGradient
from AlgorithmsLib.DQN import DQN
from AlgorithmsLib.PPO import PPO
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from runner import Runner
import numpy as np
import tensorflow as tf


Epoch = 5000
MAXSTEP = 1e5


def PG_brain():
    config = json.load(open("config.json", 'r'))
    red_team_object_list = config["red_team"]
    blue_team_object_list = config["blue_team"]

    red_team = RedTeam(red_team_object_list)
    blue_team = BlueTeam(blue_team_object_list)
    simulator = EMCSimulator(1400, 800, blue_team, red_team)

    PG = PolicyGradient(observation_dim=20, action_dim=3)

    for i_eposide in range(Epoch):
        obs = simulator.reset()
        obs = torch.FloatTensor(obs)

        step = 0
        while True:
            step += 1
            simulator.render()
            action = PG.choose_action(obs)
            red_action_list = [[action], [1, 1, 1]]       # [[Attack UAV1], [Jamming UAV1, Jamming UAV2, Jamming UAV3]]
            blue_action_list = []                         # Not set yet
            obs_, r, done = simulator.step(red_action_list, blue_action_list)
            PG.store_transition(obs, r, action)

            if done or step > MAXSTEP:
                ep_r = sum(PG.ep_r)
                loss = PG.learn()
                print("Episode: %d | Reward: %d | Loss: %.4f" % (i_eposide, ep_r, loss.item()))
                break
            obs = torch.FloatTensor(obs_)


def DQN_brain():
    render_flag = True
    config = json.load(open("config.json", 'r'))
    red_team_object_list = config["red_team"]
    blue_team_object_list = config["blue_team"]

    red_team = RedTeam(red_team_object_list)
    blue_team = BlueTeam(blue_team_object_list)
    simulator = EMCSimulator(1400, 800, blue_team, red_team, render=True)

    dqn = DQN(observation_dim=12, action_dim=3, memory_capacity=1000)

    log_name = "run/DQN_brain_no_render"
    if os.path.exists(log_name):
        shutil.rmtree(log_name)
    writer = SummaryWriter(log_name)
    writer.add_graph(dqn.evaluate_net, torch.randn(1, dqn.observation_dim))

    for i_eposide in range(Epoch):
        obs = simulator.reset()
        obs = torch.FloatTensor(obs)
        running_loss = 0
        cumulative_reward = 0
        step = 0
        while True:
            step += 1
            if render_flag:
                simulator.render()
            action = dqn.choose_action(obs)
            red_action_list = [[action], [1, 1, 1]]       # [[Attack UAV1], [Jamming UAV1, Jamming UAV2, Jamming UAV3]]
            blue_action_list = []                         # Not set yet
            obs_, r, done = simulator.step(red_action_list, blue_action_list)
            dqn.store_transition(obs, action, r, obs_)

            # It means uav has arrived at enemy's command
            if r == 500:
                # render_flag = True
                torch.save(dqn.evaluate_net, "./models/dqn/evaluate_net_%d.pth" % i_eposide)
                torch.save(dqn.target_net, "./models/dqn/target_net_%d.pth" % i_eposide)
                print("\n -> Model has saved at: './models/dqn/xx.pth'\n")


            cumulative_reward += r
            if dqn.point > dqn.memory_capacity:
                loss = dqn.learn()
                running_loss += loss
                if done or step > MAXSTEP:
                    writer.add_scalar("training/Loss", running_loss / step, dqn.learn_step)
                    writer.add_scalar("training/Reward", cumulative_reward, dqn.learn_step)
                    writer.add_scalar("training/Exploration", dqn.epsilon, dqn.learn_step)
                    print("\n - Episode: %d Cumulative Reward: %.2f\n" % (i_eposide, cumulative_reward))
                    break
                elif step % 100 == 99:
                    print("Episode: %d| Global Step: %d| Loss:  %.4f, Reward: %.2f, Exploration: %.4f" % (i_eposide, dqn.learn_step, running_loss / step, cumulative_reward, dqn.epsilon))
            else:
                print("\rCollecting experience: %d / %d..." % (dqn.point, dqn.memory_capacity), end='')

            if done:
                break
            obs = torch.FloatTensor(obs_)


def PPO_brain():
    # build the environment
    render_flag = True
    config = json.load(open("config.json", 'r'))
    red_team_object_list = config["red_team"]
    blue_team_object_list = config["blue_team"]
    red_team = RedTeam(red_team_object_list)
    blue_team = BlueTeam(blue_team_object_list)
    simulator = EMCSimulator(1400, 800, blue_team, red_team, render=True)

    # build the graph
    mean_r = tf.Variable(0, dtype=tf.float32, name= 'mean_return', trainable=False)
    mean_l = tf.Variable(0, dtype=tf.float32, name='mean_length', trainable=False)
    mean_pl = tf.Variable(0, dtype=tf.float32, name='mean_policy_loss', trainable=False)
    mean_vl = tf.Variable(0, dtype=tf.float32, name='mean_value_loss', trainable=False)
    mean_ent = tf.Variable(0, dtype=tf.float32, name='mean_entropy', trainable=False)
    mean_kl = tf.Variable(0, dtype=tf.float32, name='mean_kl_divergence', trainable=False)
    mean_cf = tf.Variable(0, dtype=tf.float32, name='mean_clip_fraction', trainable=False)
    ppo = PPO(observation_dim=12, action_dim=3, vf_coef=0.5, ent_coef=0.1, max_grad_norm=0.5)
    runner = Runner(env=simulator, model=ppo, nsteps=4096, gamma=0.99, lam=0.95, render=render_flag)

    # setup the saver and logger
    sess = tf.get_default_session()
    saver = tf.train.Saver(max_to_keep = 50)
    tf.summary.scalar('mean_return', mean_r)
    tf.summary.scalar('mean_length', mean_l)
    tf.summary.scalar('mean_policy_loss', mean_pl)
    tf.summary.scalar('mean_value_loss', mean_vl)
    tf.summary.scalar('mean_entropy', mean_ent)
    tf.summary.scalar('mean_kl_divergence', mean_kl)
    tf.summary.scalar('mean_clip_fraction', mean_cf)
    merged_summary_op = tf.summary.merge_all()
    summary_wirter = tf.summary.FileWriter('run/ppo', sess.graph)

    # configure the training hyperparameters
    lr = 3e-4
    cr = 0.2
    nbatch = 4096
    nminibatch = 64
    noptepochs = 4
    nupdates = 2500
    nbatch_train = int(nbatch / nminibatch)
    log_interval = 5
    save_interval = 50
    total_length = []
    total_reward = []

    # start the training procedure
    for update in range(1, nupdates + 1):
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr * frac
        crnow = cr * frac

        # stepping the environment
        b_obs, b_ret, b_act, b_opv, b_olp, t_len, t_rew = runner.run()
        total_length += t_len
        total_reward += t_rew

        # updating the network
        inds = np.arange(nbatch)
        stats = []
        for _ in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (b_obs, b_act, b_ret, b_opv, b_olp))
                stats.append(ppo.train(lrnow, crnow, *slices))

        # mean stats of the current update
        mean_stats = np.mean(stats, axis=0)
        up1 = tf.assign(mean_pl, mean_stats[0])
        up2 = tf.assign(mean_vl, mean_stats[1])
        up3 = tf.assign(mean_ent, mean_stats[2])
        up4 = tf.assign(mean_kl, mean_stats[3])
        up5 = tf.assign(mean_cf, mean_stats[4])
        sess.run([up1, up2, up3, up4,up5])

        # logging with fixed interval
        if(update % log_interval == 0):
            t_len_mean = np.mean(total_length)
            t_rew_mean = np.mean(total_reward)
            up6 = tf.assign(mean_l, t_len_mean)
            up7 = tf.assign(mean_r, t_rew_mean)
            sess.run([up6, up7])
            summary_str = sess.run(merged_summary_op)
            summary_wirter.add_summary(summary_str, update)
            total_length = []
            total_reward = []

        # saving with fixed interval
        if(update % save_interval == 0):
            saver.save(sess, 'models/ppo/ppo', global_step=update)


def PPO_test():
    render_flag = True
    config = json.load(open("config.json", 'r'))
    red_team_object_list = config["red_team"]
    blue_team_object_list = config["blue_team"]
    red_team = RedTeam(red_team_object_list)
    blue_team = BlueTeam(blue_team_object_list)
    simulator = EMCSimulator(1400, 800, blue_team, red_team, render=True)

    ppo = PPO(observation_dim=12, action_dim=3, vf_coef=0.5, ent_coef=0.1, max_grad_norm=0.5)
    runner = Runner(env=simulator, model=ppo, nsteps=4096, gamma=0.99, lam=0.95, render=render_flag)

    saver = tf.train.Saver()
    sess = tf.get_default_session()
    saver.restore(sess, "models/ppo/ppo-2500")

    rew_list = []
    for _ in range(300):
        _, _, _, _, _, _, rew = runner.run()
        rew_list += rew
    print('Test results: mean reward {} in {} episodes'.format(np.mean(rew_list), len(rew_list)))


if __name__ == '__main__':
    # PG_brain()
    # DQN_brain()
    # PPO_brain()
    PPO_test()
