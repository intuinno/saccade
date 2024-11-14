import torch
import torch.nn as nn
import torch.optim as optim
import gym

# Hyperparameters
learning_rate = 1e-3
gamma = 0.99  # Discount factor
lambda_ = 0.95  # Lambda for lambda-return
n_episodes = 1000
env_name = "CartPole-v1"


# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(128, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        policy_dist = self.actor(x)
        value = self.critic(x)
        return policy_dist, value


# Lambda-return computation
def lambda_return(rewards, values, gamma, lambda_):
    returns = []
    G = values[-1].item()  # Bootstrapped final value
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * ((1 - lambda_) * values[t].item() + lambda_ * G)
        returns.insert(0, G)
    return torch.tensor(returns)


# Training Loop
def train(env, model, optimizer):
    for episode in range(n_episodes):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []

        done = False
        while not done:
            state = torch.tensor(state, dtype=torch.float32)
            policy_dist, value = model(state)
            action = torch.multinomial(policy_dist, 1).item()

            next_state, reward, done, _, _ = env.step(action)

            log_prob = torch.log(policy_dist[action])
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        # Add final value as bootstrap (if non-terminal)
        final_value = (
            torch.tensor([0.0])
            if done
            else model(torch.tensor(next_state, dtype=torch.float32))[1]
        )
        values.append(final_value)

        # Compute λ-returns
        lambda_returns = lambda_return(rewards, values, gamma, lambda_)

        # Compute advantages
        values = torch.stack(
            values[:-1]
        ).squeeze()  # Remove last bootstrap value for consistency
        advantages = lambda_returns - values.detach()

        # Compute actor and critic losses
        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + critic_loss

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode+1}, Loss: {loss.item()}, Total Reward: {sum(rewards)}"
            )


# Main
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(env, model, optimizer)
env.close()
