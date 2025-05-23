import numpy as np
import pandas as pd
import random

class ISamplerAgent:
    def __init__(self, delta=0.5, kappa=10, eta=0.1):
        self.delta = delta
        self.kappa = kappa
        self.eta = eta
        self.memory = []
        self.last_choice = None
        self.last_reward = None

    def surprise(self, last_reward):
        if not self.memory:
            return 1.0
        rewards = [r for _, r in self.memory]
        if not rewards:
            return 1.0
        mean = np.mean(rewards)
        range_ = max(rewards) - min(rewards)
        if range_ == 0:
            return 1.0
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
        means = []
        for i in [0, 1]:
            if avg_rewards[i]:
                means.append(np.mean(avg_rewards[i]))
            else:
                means.append(0.0)
        return int(means[1] > means[0])

    def update(self, choice, reward):
        self.memory.append((choice, reward))
        self.last_choice = choice
        self.last_reward = reward


def simulate_task(prior_score, a_prob, a1, a2, b_prob, b1, b2, forgone, n_participants, n_trials):
    results = {
        'arate1': [], 'arate2': [], 'arate3': [], 'arate4': [],
        'afoka': [], 'afrega': [], 'afregb': [], 'afokb': []
    }

    def get_reward(choice):
        if choice == 0:
            return a1 if random.random() < a_prob else a2
        else:
            return b1 if random.random() < b_prob else b2

    for _ in range(n_participants):
        agent = ISamplerAgent()
        block_choices = [[] for _ in range(4)]
        condition_stats = {'afoka': [], 'afrega': [], 'afregb': [], 'afokb': []}

        for t in range(n_trials):
            block = t // (n_trials // 4)
            choice = agent.choose(t, prior_score)
            reward = get_reward(choice)

            if forgone:
                other_reward = get_reward(1 - choice)
                agent.update(choice, reward)
                agent.update(1 - choice, other_reward)
            else:
                agent.update(choice, reward)

            block_choices[block].append(choice)

            if reward > 0:
                condition_stats['afoka'].append(choice)
                condition_stats['afregb'].append(1 - choice)
            else:
                condition_stats['afrega'].append(choice)
                condition_stats['afokb'].append(1 - choice)

        # Calculate block averages
        for i in range(4):
            if block_choices[i]:  # Only calculate mean if there are choices
                results[f'arate{i+1}'].append(np.mean(block_choices[i]))
            else:
                results[f'arate{i+1}'].append(0.0)  # Default to 0 if no choices

        # Calculate condition averages
        for k in condition_stats:
            if condition_stats[k]:  # Only calculate mean if there are stats
                results[k].append(np.mean(condition_stats[k]))
            else:
                results[k].append(0.0)  # Default to 0 if no stats

    # Calculate final averages, ensuring no nan values
    final_results = {}
    for k, v in results.items():
        if v:  # If there are any values
            final_results[k] = np.mean(v)
        else:
            final_results[k] = 0.0  # Default to 0 if no values

    return final_results


def evaluate_against_dataset(excel_path):
    df = pd.read_excel(excel_path)
    target_cols = ['arate1', 'arate2', 'arate3', 'arate4', 'afoka', 'afrega', 'afregb', 'afokb']
    df_clean = df[df['ChatGPTo'].notna()]
    for col in target_cols:
        df_clean = df_clean[df_clean[col].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
    df_clean[target_cols] = df_clean[target_cols].astype(float)

    predicted_results = []
    for _, row in df_clean.iterrows():
        sim_result = simulate_task(
            prior_score=float(row['ChatGPTo']),
            a_prob=float(row['pa1']), a1=float(row['a1']), a2=float(row['a2']),
            b_prob=float(row['pb1']), b1=float(row['b1']), b2=float(row['b2']),
            forgone=bool(row['forgone']), n_participants=int(row['n']), n_trials=100
        )
        predicted_results.append(sim_result)

    pred_df = pd.DataFrame(predicted_results)
    gt_df = df_clean[target_cols].reset_index(drop=True)
    comparison_df = pd.concat([pred_df.add_prefix('pred_'), gt_df.add_prefix('gt_')], axis=1)
    comparison_df['MSE'] = ((pred_df - gt_df) ** 2).mean(axis=1)

    print("\n--- Summary ---")
    print(comparison_df.mean())
    print("\n--- Detailed Comparison ---")
    print(comparison_df.head())

    return comparison_df

if __name__ == "__main__":
    file_path = "Training 2025.04.22DM.xlsx"  # Adjust path if needed
    # evaluate_against_dataset(file_path)

    sim_result = simulate_task(
            prior_score=0.215,
            a_prob=1, a1=0, a2=0,
            b_prob=0.1, b1=10, b2=-1,
            forgone=True, n_participants=300, n_trials=100
        )
    
    print("\nSimulation Results:")
    print("-----------------")
    print("Block Averages:")
    print(f"  Block 1: {sim_result['arate1']:.3f}")
    print(f"  Block 2: {sim_result['arate2']:.3f}")
    print(f"  Block 3: {sim_result['arate3']:.3f}")
    print(f"  Block 4: {sim_result['arate4']:.3f}")
    print("\nCondition Averages:")
    print(f"  After OK A: {sim_result['afoka']:.3f}")
    print(f"  After Regret A: {sim_result['afrega']:.3f}")
    print(f"  After Regret B: {sim_result['afregb']:.3f}")
    print(f"  After OK B: {sim_result['afokb']:.3f}")