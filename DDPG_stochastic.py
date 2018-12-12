import tensorflow as tf
import numpy as np
import rospy
import random
import time
import copy
import os.path
import pickle

from StageWorld import StageWorld
from PrioritizedBuffer import Memory
from noise import Noise
from reward import Reward
from actor import ActorNetwork
from critic import CriticNetwork

import matplotlib.pyplot as plt
import pylab as plt
import matplotlib.colors as colors

MAX_EPISODES = 50000
NOISE_MAX_STEP = 100000
# Noise parameters
DELTA = 0.1 # The rate of change (time)
SIGMA = 0.1 # Volatility of the stochastic processes
OU_A = 0.2 # The rate of mean reversion
OU_MU = 0. # The long run average interest rate

REWARD_FACTOR = 0.1 # Total episode reward factor
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

LASER_BEAM = 40
LASER_HIST = 3
LASER_MIN = 1
ACTION = 2
TARGET = 2
SPEED = 2
SWITCH = 3

DELAY_SWITCH_TRAINING = 1000
SWITCH_DECAY_START = 70000
SWITCH_DECAY_END = 80000
RANDOM_SWITCH = False
CONTROL_VARIANCE = True

# ===========================
#   Utility Parameters
# ===========================
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 32

GAME = 'StageWorld'

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, noise, reward, discrete, action_bound):
    # Set up summary writer
    summary_writer = tf.summary.FileWriter("ddpg_summary", sess.graph)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    print('checkpoint:', checkpoint)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    if os.path.isfile('PrioritizedMem.p'):
        memory = pickle.load(open('PrioritizedMem.p', "rb"))
        print 'Load Memory'
        LOAD_FLAG = True
    else:
        memory = Memory(capacity=BUFFER_SIZE)
        print 'Create Memory'
        LOAD_FLAG = False


    # # plot settings
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # im = ax.imshow(env.map, aspect='auto', cmap='hot', vmin=0., vmax=1.5)
    # plt.show(block=False)

    # Initialize noise
    ou_level = [0., 0.]

    rate = rospy.Rate(5)
    loop_time = time.time()
    last_loop_time = loop_time
    i = 0

    s1 = env.GetLaserObservation()
    s_1 = np.stack((s1, s1, s1), axis=1)

    T = 0.
    ep_r_window = []
    for i in range(MAX_EPISODES):
        env.ResetWorld()
        env.GenerateTargetPoint()
        # print 'Target: (%.4f, %.4f)' % (env.target_point[0], env.target_point[1])
        target_distance = copy.deepcopy(env.distance)
        ep_reward = 0.
        ep_ave_max_q = 0.
        loop_time_buf = []
        terminal = False

        ep_state = []
        ep_switch = []
        action = [0., 0.]
        j = 0
        ep_reward = 0
        ep_ave_max_q = 0
        ep_PID_count = 0.
        ep_OA_count = 0.
        switch_index = 0.
        ep_metric = 0.
        while not terminal and not rospy.is_shutdown():
            s1 = env.GetLaserObservation()
            s_1 = np.append(np.reshape(s1, (LASER_BEAM, 1)), s_1[:, :(LASER_HIST - 1)], axis=1)
            s__1 = np.reshape(s_1, (LASER_BEAM * LASER_HIST))
            target1 = env.GetLocalTarget()
            speed1 = env.GetSelfSpeed()
            state1 = np.concatenate([s__1, speed1, target1], axis=0)
            [x, y, theta] =  env.GetSelfStateGT()
            map_img = env.RenderMap([[0, 0], env.target_point])
            
            r, terminal, result, metric = env.GetRewardAndTerminate(j, switch_index)
            ep_reward *= GAMMA
            ep_reward += r
            ep_metric += metric

            if j > 0 :
                memory.store_transition(state, a[0], r, terminal, state1) 

            j += 1
            state = state1

            a = actor.predict(np.reshape(state, (1, actor.s_dim)))

            logits, switch_a_tmp = critic.predict_switch(np.reshape(state, (1, actor.s_dim))) 
            if j%(LASER_HIST+2) == 1:
                switch_a = switch_a_tmp
                ep_state.append(state)
                ep_switch.append(switch_a[0])

            if T < SWITCH_DECAY_START:
                switch_flag = True
            elif np.random.rand() > (T - SWITCH_DECAY_START) / (SWITCH_DECAY_END - SWITCH_DECAY_START):
                switch_flag = True
            else:
                switch_flag = False

            if switch_flag:
                if RANDOM_SWITCH:
                    switch_a = [np.random.randint(SWITCH)]
                if switch_a[0] == 1:
                    a = env.PIDController(action_bound)
                    ep_PID_count += 1.
                elif switch_a[0] == 2:
                    a = env.OAController(action_bound, action)
                    ep_OA_count += 1. 

            if T < NOISE_MAX_STEP:
                ou_level = [noise.ornstein_uhlenbeck_level(ou_level[0]),\
                            noise.ornstein_uhlenbeck_level(ou_level[1])]
                a = a + ou_level

            action = a[0]
            if  action[0] <= 0.05:
                action[0] = 0.05
            elif action[0] > action_bound[0]:
            	action[0] = action_bound[0]
           	if action[1] < -action_bound[1]:
           		action[1] = -action_bound[1]
           	elif action[1] > action_bound[1]:
           		action[1] = action_bound[1]

            env.Control(action)

            # # plot
            # if j == 1:
            #     im.set_array(map_img)
            #     fig.canvas.draw()

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if T == BUFFER_SIZE+5000 and not LOAD_FLAG:
                pickle.dump(memory, open("PrioritizedMem.p", "wb"))

            if T > BUFFER_SIZE+5000 or LOAD_FLAG:
                tree_idx, batch_memory, ISWeights = memory.sample(MINIBATCH_SIZE)
                s_batch = batch_memory[:, :critic.s_dim]
                a_batch = batch_memory[:, critic.s_dim:critic.s_dim+2]
                r_batch = batch_memory[:, critic.s_dim+2]
                s2_batch = batch_memory[:, -critic.s_dim:]
                t_batch = batch_memory[:, critic.s_dim+3]
                # Calculate targets
                # critic
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))
                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # switch
                if terminal and not RANDOM_SWITCH:
                    _ = critic.switch_train(ep_state, ep_switch, ep_reward)

                # Update the critic given the targets
                predicted_q_value, abs_errors, _ = critic.train(s_batch, a_batch,\
                                                    np.reshape(y_i, (MINIBATCH_SIZE, 1)), \
                                                    ISWeights)

                ep_ave_max_q += np.amax(predicted_q_value)
                memory.batch_update(tree_idx, abs_errors)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            last_loop_time = loop_time
            loop_time = time.time()
            loop_time_buf.append(loop_time - last_loop_time)
 
            pose = env.GetSelfStateGT()
            file_string = str(i) + ', ' + str(j) + ', ' + str(pose[0]) + ', ' + str(pose[1]) + \
                          ', ' + str(switch_index) + '\n'
            # f = open('./epi_t_pose_switch.txt', 'a')
            # f.write(file_string)
            T += 1
            rate.sleep()


        summary = tf.Summary()
        summary.value.add(tag='Reward', simple_value=float(ep_reward))
        summary.value.add(tag='Metric', simple_value=float(ep_metric))
        summary.value.add(tag='Qmax', simple_value=float(ep_ave_max_q / float(j)))
        summary.value.add(tag='PIDrate', simple_value=float(ep_PID_count / float(j)))
        summary.value.add(tag='OArate', simple_value=float(ep_OA_count / float(j)))
        summary.value.add(tag='POLICYrate', simple_value=float(j - ep_OA_count - ep_PID_count) / float(j))
        summary_writer.add_summary(summary, T)
        summary_writer.flush()

        # if i > 0 and i % 1000 == 0 :
        #     saver.save(sess, 'saved_networks/' + GAME + '-ddpg', global_step = i) 

        print '| Reward: %.2f' % ep_reward, "| Episode:", i, \
        '| Qmax: %.2f' % (ep_ave_max_q / float(j)), \
        "| LoopTime: %.4f" % (np.mean(loop_time_buf)), "| T:", T, '\n', \
        '| PID: %.2f' % (ep_PID_count / float(j)), '| OA: %.2f' % (ep_OA_count / float(j))
        
        if T > 100000:
            saver.save(sess, 'saved_networks/' + GAME + '-ddpg', global_step = i) 
            # rospy.signal_shutdown("Time up") 
            break   


def main(_):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        env = StageWorld(LASER_BEAM)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        state_dim = LASER_BEAM * LASER_HIST + SPEED + TARGET 

        action_dim = ACTION
        action_bound = [0.5, np.pi/3]
        switch_dim = SWITCH

        discrete = False
        print('Continuous Action Space')
        with tf.name_scope("Actor"):
            actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                                  ACTOR_LEARNING_RATE, TAU)
        with tf.name_scope("Critic"):
            critic = CriticNetwork(sess, state_dim, action_dim, switch_dim,
                                    CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(),
                                    baseline_rate=10., control_variance_flag=CONTROL_VARIANCE)

        noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
        reward = Reward(REWARD_FACTOR, GAMMA)

        try:
            train(sess, env, actor, critic, noise, reward, discrete, action_bound)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    tf.app.run()
