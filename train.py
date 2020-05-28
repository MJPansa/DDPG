import numpy as np
import torch as T
from models import DDPGActor, DDPGCritic
from utils import DDPGExperienceBuffer
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
from torch.distributions import Normal
import wandb
import sys
import gym

wandb.init(project="actor_critic_continuous")

env = gym.make('LunarLanderContinuous-v2')

args = {'obs_space': 8,
        'action_space': 2,
        'n_hidden': 300,
        'bs': 256,
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'device': 'cuda:0',
        'gamma': .999,
        'noise_factor': 0.01,
        'noise_decay': 0.999,
        'buffer_size': 1_000_000,
        'critic_weight': .5,
        'episodes': 5000,
        'clipping': 2.,
        'buffer_threshold': .05,
        'train_every_n': 1,
        'model_update_every_n': 1,
        'tau': 0.01,
        'clip_grad': 0.1,
        'exploration_steps': 10000
        }

print(args)

actor = DDPGActor(args['obs_space'], args['action_space'], args['n_hidden'], args['lr_actor'], args['device'])
critic = DDPGCritic(args['obs_space'], args['action_space'], args['n_hidden'], args['lr_critic'], args['device'])
actor_target = DDPGActor(args['obs_space'], args['action_space'], args['n_hidden'], args['lr_actor'], args['device'])
critic_target = DDPGCritic(args['obs_space'], args['action_space'], args['n_hidden'], args['lr_critic'], args['device'])
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

exp = DDPGExperienceBuffer(args['buffer_size'], args['bs'], args['buffer_threshold'], args['device'])

wandb.watch(actor)
wandb.watch(critic)
step = 0
for episode in range(args['episodes']):

    stats = {
        'rewards': 0.,
        'actor_loss': 0.,
        'critic_loss': 0.,
        'loss': 0.
    }
    exp_cache = []
    done = False
    state = env.reset()
    while not done:
        actor.eval()
        exp_cache.append(T.Tensor(state))
        # print(state.shape)
        if step < args['exploration_steps']:
            action = np.random.uniform(-1, 1, size=2)
        else:
            action = actor(state)
            # print(actions.size())

            action = action.squeeze().detach().cpu().numpy()

            noise = np.random.randn(args['action_space']) * args['noise_factor']
            action = np.clip(action + noise, -1., 1.)
        exp_cache.append(T.Tensor(action))

        next_state, reward, done, _ = env.step(action)

        stats['rewards'] += reward
        args['noise_factor'] *= args['noise_decay']
        step += 1

        exp_cache.append(T.Tensor([reward]))
        exp_cache.append(T.Tensor([done]))
        exp_cache.append(T.Tensor(next_state))
        exp.add(*exp_cache)
        exp_cache.clear()

        state = next_state.copy()

    if exp.threshold:
        actor.train()
        critic.train()
        critic_target.train()
        actor_target.train()

        o, a, r, d, o2 = exp.draw()

        q = critic(o, a)

        q_target = critic_target(o2, actor_target(o2))
        q_target = r + args['gamma'] * q_target * (1 - d)

        critic_loss = ((q - q_target.detach()) ** 2).mean()

        critic.optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(critic.parameters(), args['clip_grad'])
        critic.optimizer.step()

        a_ = actor(o)
        actor_loss = -1 * critic(o, a_).mean()

        actor.optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(actor.parameters(), args['clip_grad'])
        actor.optimizer.step()

        stats['actor_loss'] += actor_loss
        stats['critic_loss'] += critic_loss
        stats['loss'] += (actor_loss + critic_loss)

        # if step % args['model_update_every_n'] == 0:

        for target_param, local_param in zip(actor_target.parameters(), actor.parameters()):
            target_param.data.copy_(args['tau'] * local_param.data + (1.0 - args['tau']) * target_param.data)

        for target_param, local_param in zip(critic_target.parameters(), critic.parameters()):
            target_param.data.copy_(args['tau'] * local_param.data + (1.0 - args['tau']) * target_param.data)

    print(f'episode {episode}:')
    print(f'rewards: {stats["rewards"]:.5f}')
    print(f'actor loss: {stats["actor_loss"]:.5f}')
    print(f'critic loss: {stats["critic_loss"]:.5f}')
    print(f'noise factor: {args["noise_factor"]:.5f}')
    print(f'buffer size: {len(exp)}\n')
    wandb.log(stats)
