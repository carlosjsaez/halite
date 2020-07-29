from kaggle_environments.envs.halite.helpers import *
from kaggle_environments import evaluate, make
import gym
import tensorflow as tf
import numpy as np
import pandas as pd
from random import sample
from matplotlib.pyplot import plot

env = make('halite', debug = True)
env_config = env.configuration

training_env = env.train([None, "random"])
env.render(mode="ipython",width=800, height=600)

obs = training_env.reset()
board = Board(obs, env_config)
size = env_config['size']**2
input_size = size
init_input = tf.keras.initializers.GlorotNormal()
init_output = tf.keras.initializers.RandomUniform(0, 1)
tf.keras.backend.set_floatx('float64')

# NETWORKS DEFINITION

network_ship = tf.keras.Sequential()
network_ship.add(tf.keras.layers.Dense(input_size, kernel_initializer = init_input))
network_ship.add(tf.keras.layers.Dropout(0.3))
network_ship.add(tf.keras.layers.Dense(256, kernel_initializer = init_input))
network_ship.add(tf.keras.layers.Dropout(0.3))
network_ship.add(tf.keras.layers.Dense(5, kernel_initializer = init_output))

network_ship.build(input_shape = (None, input_size))
network_ship.summary()

network_shipyard = tf.keras.Sequential()
network_shipyard.add(tf.keras.layers.Dense(input_size))
network_shipyard.add(tf.keras.layers.Dropout(0.3))
network_shipyard.add(tf.keras.layers.Dense(256))
network_shipyard.add(tf.keras.layers.Dropout(0.3))
network_shipyard.add(tf.keras.layers.Dense(1))

network_shipyard.build(input_shape = (None, input_size))
network_shipyard.summary()

network_critic = tf.keras.Sequential()
network_critic.add(tf.keras.layers.Dense(input_size))
network_critic.add(tf.keras.layers.Dropout(0.3))
network_critic.add(tf.keras.layers.Dense(256))
network_critic.add(tf.keras.layers.Dropout(0.3))
network_critic.add(tf.keras.layers.Dense(1))

network_critic.build(input_shape = (None, input_size))
network_critic.summary()

# INITIALISE

activation = 'relu'
epochs = 1000
eps = 0.1

def logp(logits, action, depth):
    logp_all = tf.nn.log_softmax(tf.reshape(logits, (-1,depth)))
    one_hot = tf.one_hot(action, depth, dtype = 'float64')
    logp = tf.reduce_sum(one_hot * logp_all, axis=-1)
    return logp

def entropy(logits):
    """
        Entropy term for more randomness which means more exploration \n
        Based on OpenAI Baselines implementation
    """
    # Need to check this calculations
    a0 = logits - tf.reduce_max(logits, axis= -1, keepdims=True)
    exp_a0 = tf.exp(a0)
    z0 = tf.reduce_sum(exp_a0, axis= -1, keepdims=True)
    p0 = exp_a0 / z0
    entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis= -1)
    return entropy

def adv_rets(rews, values,dones, gamma = 0.6, lamb = 0.4):

    advs = np.zeros_like(rews)
    last_gae_lam = 0
    steps = len(values)

    for t in reversed(range(steps)):
        if t == steps - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_values = values[-1]
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

    delta = rews[t] + gamma * next_values * next_non_terminal - values[t]
    advs[t] = last_gae_lam = delta + gamma * lamb * next_non_terminal * last_gae_lam

    returns = advs + values  # ADV = RETURNS - VALUES
    advs = (advs - advs.mean()) / (advs.std())  # Normalize ADVs

    return advs, returns

# loss and optimiser

def ppo(epoch_hist):

    lr = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr, clipvalue = (-1,1))
    train_iters = 4
    n_batch = len(epoch_hist['values'])
    n_minibatch = int(n_batch/10)

    for i in range(train_iters):
        inds = np.arange(n_batch)
        np.random.shuffle(inds)
        # Minibatch random shuffle for the updates

        for start in range(0, n_batch, n_minibatch):
            # Compute losses
            slices = inds[start:(start+n_minibatch)]


def policy_loss(epoch_hist, idxs, network = network_ship):

    min_adv = 1
    ent_coef = 0.1

    state = np.array(epoch_hist['board'])[idxs]
    actions = np.array(epoch_hist['actions'])[idxs]
    logp_old = np.array(epoch_hist['logp'])[idxs]
    advs = np.array(epoch_hist['advs'])[idxs]

    pi = network_ship(state)


    logp_pi = logp(pi, actions, 5)  # PPO Objective
    ratio = tf.exp(logp_pi - logp_old.reshape(-1))

    clipped_loss = -tf.reduce_mean(
        tf.math.minimum(ratio * advs, min_adv))  # Policy Loss = loss_clipped + entropy_loss * entropy_coef
    # losses have negative sign for maximizing via backprop

    entropy_pi = entropy(pi)  # Entropy loss - Categorical Policy --> returns entropy based on logits
    entropy_loss = -tf.reduce_mean(entropy_pi)

    pi_loss = clipped_loss + entropy_loss * ent_coef  # Policy Loss

    # approx_kl = tf.reduce_mean(logp_old - logp)  # Approximated  Kullback Leibler Divergence from OLD and NEW Policy
    # approx_ent = tf.reduce_mean(-logp)

    # total_loss = pi_loss + v_loss * v_coef  # Total Loss = pi_loss + value loss * v_coef


    return pi_loss

def critic_loss(epoch_hist, idxs):

    values = values_new = network_critic(state)
    rets = np.array(epoch_hist['rets'])[idxs]
    v_loss = 0.5 * tf.reduce_mean(tf.square(rets - values))

    return v_loss


optim = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1)

# TRAINING
# Plot rewards to check improvements
plot_rews = []

for epoch in range(epochs):

    boards_ = []
    actions_ = []
    values_ = []
    logp_ = []
    entropy_ = []
    rews_ = []
    dones_ = []

    obs = training_env.reset()
    board = Board(obs, env_config)
    size = env_config['size'] ** 2
    # input_size = size
    # network
    # loss and optimiser

    done = False
    while done == False:
        state = np.array(board.observation['halite']).reshape(1,-1)
        step = board.observation['step']
        cur_halite = 0
        for ship in board.current_player.ships[0:1]:
            # ship = board.ships['0-1']
            pos = ship.position
            ships_state = relative_state(state, pos, size)
            logits = network_ship(ships_state)
            int_action_ship = eps_greedy(eps, logits) # if step > 0 else 1
            cur_halite += ship._halite
            # Log P calculations
            logp_ship = logp(logits, int_action_ship, 5)
            entropy_ship = entropy(logits)
            ship.next_action = convert_action_ship(ship.id,
                                                   obs,
                                                   int_action_ship # if step > 0 else 5
                                                   )

        dict_ships = {ship.id: ship.next_action for ship in board.current_player.ships if ship.next_action is not None}

        for shipyard in board.current_player.shipyards[0:1]:
            # ship = board.ships['0-1']

            logits = network_shipyard(state)
            int_action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1).numpy()[0]
            # Log P calculations
            logp_shipyard = logp(logits, int_action, 2)
            entropy_shipyard = entropy(logits)
            shipyard.next_action = convert_action_shipyard(shipyard.id,
                                                           obs,
                                                           1 # if step == 1 else None
                                                           )
        dict_shipyards = {shipyard.id : shipyard.next_action for shipyard in board.current_player.shipyards if shipyard.next_action is not None}
        dict_actions = dict(dict_ships, **dict_shipyards)

        values = network_critic(state)

        new_obs = training_env.step(dict_actions)
        done = new_obs[2]
        board = Board(new_obs[0], env_config)

        rewards = new_obs[1] + sum([ship._halite for ship in board.current_player.ships]) - cur_halite

        boards_.append(ships_state)
        actions_.append(int_action_ship)
        values_.append(values)
        logp_.append(logp_ship)
        entropy_.append(entropy_ship)
        rews_.append(rewards)
        dones_.append(done)
        env.render()

        print(dict_actions, step, rewards,cur_halite, pos)

    epoch_hist = {'board': boards_,
                       'actions': actions_,
                       'values': values_,
                       'logp': logp_,
                       'entropy': entropy_,
                       'rews': rews_,
                       'dones': dones_}

    # Calculate Adcantages (GAE?)

    epoch_hist['advs'], epoch_hist['rets'] = adv_rets(epoch_hist['rews'], epoch_hist['values'], epoch_hist['dones'])


    # Batch losses and update

    n_batches = 10
    ratio_batch = 0.2
    batch_size = int(len(epoch_hist['board'])*ratio_batch)

    for batch in range(n_batches):

        print('Updating batch', batch+1,'out of ', n_batches)

        idxs = sample(range(len(epoch_hist['board'])), batch_size)

        # Update
        for network, func_loss in zip([network_ship, network_critic],[policy_loss, critic_loss]):
            with tf.GradientTape() as tape:
                loss = func_loss(epoch_hist, idxs)
                weights = network.trainable_variables
                tape.watch(weights)

            grads = tape.gradient(loss, weights)
            grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)               # clip gradients for slight updates
            optim.apply_gradients(zip(grads, weights))
            # tape.reset()
    if epoch % 10 == 0:
        network_ship.save('//actor')
        network_critic.save('//critic')

    plot_rews.append(np.sum(epoch_hist['rews']))

plot(plot_rews);

