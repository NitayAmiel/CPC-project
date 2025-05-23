import numpy as np
import random

class ISamplerAgent:
    def __init__(self, delta=0.5, kappa=5, eta=0.1):
        self.delta = delta  # reliance on prior
        self.kappa = kappa  # experience sampling window
        self.eta = eta      # inertia decay threshold
        self.memory = []
        self.last_choice = None
        self.last_reward = None

    def surprise(self, last_reward):
        if not self.memory:
            return 1.0
        rewards = [r for _, r in self.memory]
        mean = np.mean(rewards)
        range_ = max(rewards) - min(rewards) or 1.0
        return abs(mean - last_reward) / range_

    def choose(self, t, prior_score):
        if t == 0 or random.random() < self.delta:
            return int(prior_score > 0.5)
        if random.random() > self.eta * self.surprise(self.last_reward):
            return self.last_choice
        sample = random.sample(self.memory, min(self.kappa, len(self.memory)))
        avg_rewards = {0: [], 1: []}
        for c, r in sample:
            avg_rewards[c].append(r)
        means = [np.mean(avg_rewards[i]) if avg_rewards[i] else 0 for i in [0, 1]]
        return int(means[1] > means[0])

    def update(self, choice, reward):
        self.memory.append((choice, reward))
        self.last_choice = choice
        self.last_reward = reward


def evaluate_agent(agent_cls, dataset):
    total_correct = 0
    total_choices = 0
    for entry in dataset:
        agent = agent_cls()
        prior_score = entry['prior_score']
        trials = entry['trials']
        for t, (true_choice, reward) in enumerate(trials):
            pred = agent.choose(t, prior_score)
            if pred == true_choice:
                total_correct += 1
            total_choices += 1
            agent.update(true_choice, reward)
    return total_correct / total_choices


def hyperparameter_search(data, delta_vals, kappa_vals, eta_vals):
    best_acc = 0
    best_params = None
    for delta in delta_vals:
        for kappa in kappa_vals:
            for eta in eta_vals:
                def agent_fn():
                    return ISamplerAgent(delta=delta, kappa=kappa, eta=eta)
                acc = evaluate_agent(agent_fn, data)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (delta, kappa, eta)
    return best_params, best_acc


def predict_choices(agent_params, prior_score, num_trials, rewards=None):
    agent = ISamplerAgent(*agent_params)
    predictions = []
    for t in range(num_trials):
        choice = agent.choose(t, prior_score)
        predictions.append(choice)
        reward = rewards[t] if rewards else 0
        agent.update(choice, reward)
    return predictions


# Example usage
if __name__ == "__main__":
    example_data = [
        {
            'prior_score': 0.76,
            'trials': [(0, 0), (1, 1), (1, -1), (0, 0)]
        },
        {
            'prior_score': 0.32,
            'trials': [(1, -1), (0, 1), (0, 0)]
        }
    ]

    deltas = [0.3, 0.5, 0.7]
    kappas = [3, 5, 10]
    etas = [0.05, 0.1, 0.2]

    best_params, best_acc = hyperparameter_search(example_data, deltas, kappas, etas)
    print(f"Best Params: {best_params}, Accuracy: {best_acc:.3f}")

    predictions = predict_choices(best_params, prior_score=0.65, num_trials=5, rewards=[1, -1, 0, 1, 1])
    print(f"Predicted Choices: {predictions}")

