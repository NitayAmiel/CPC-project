import numpy as np
import pandas as pd
import random
import math

A = 1
B = 2

def sample_from_list(lst, n, distribution="uniform", spread=3):
    """
    Samples n elements from lst based on the specified distribution.

    Parameters:
        lst (list): The list to sample from.
        n (int): Number of elements to sample.
        distribution (str): "uniform" or "half_gaussian_bell".
        spread (float): Controls steepness for half_gaussian_bell (default 3).

    Returns:
        list: Sampled elements.
    """
    if len(lst) < n:
        raise ValueError("n cannot be larger than the length of the list")

    if distribution == "uniform":
        return list(np.random.choice(lst, size=n, replace=False))

    elif distribution == "half_gaussian_bell":
        x = np.linspace(-spread, 0, len(lst))
        weights = np.exp(-x**2 / 2)
        weights /= weights.sum()
        return list(np.random.choice(lst, size=n, replace=False, p=weights))

    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

class ISamplerAgent:
    def __init__(self, delta=0.5, kappa=10, eta=0.1, padescriptor=None, distribution = "half_gaussian_bell"):
        self.delta = delta
        self.kappa = kappa
        self.eta = eta
        self.padescriptor = padescriptor
        self.distribution = distribution
        self.memory = []  # Each entry will be (choice, A_payoff, B_payoff) where choice is 1 for A, 0 for B

    def surprise(self):
        if not self.memory:
            return 0.0
        
        j = self.memory[-1][0]
        V_j_t = self.memory[-1][j]
        
        rewards = []
        for r in self.memory:
            if r[j] is not None:
                rewards.append(r[j])
        max_reward = max(rewards)
        min_reward = min(rewards)
        if max_reward == min_reward:
            return 0.0
        mean_reward = np.mean(rewards)
        return abs(mean_reward - V_j_t) / (max_reward - min_reward)
    

    def get_rand_indexes(self, num_of_idx, type = None):
        idx_list = []
        for idx in len(self.memory):
            if type == None or self.memory[idx][0] == type:
                idx_list.append(idx)
        assert(len(idx_list) != 0)
        num_of_idx = min(num_of_idx, len(idx_list))
        return sample_from_list(idx_list, num_of_idx, self.distribution)

        
    def get_sample_prob(self):
        if self.forgone:
            num_of_A = 0
            for r in self.memory:
                if r[0] == A :
                    num_of_A+= 1
            percentage_A = num_of_A / len(self.memory)
            kappa_A = math.ceil(percentage_A*self.kappa)
            kappa_B = math.ceil((1-percentage_A)*self.kappa)
            rewards_rand_index_A = self.get_rand_indexes(kappa_A, A)# indexes
            rewards_rand_index_B =  self.get_rand_indexes(kappa_B, B) # indexes
            reward_rand_A = []
            reward_rand_B = []
            for idx in rewards_rand_index_A:
                reward_rand_A.append(self.memory[idx][A])
            for idx in rewards_rand_index_B:
                reward_rand_B.append(self.memory[idx][B])
            avg_A = np.mean(reward_rand_A)
            avg_B = np.mean(reward_rand_B)
            if avg_A*avg_B <= 0:
                if avg_A > 0:
                    return 1
                else:
                    return 0
            if avg_A < 0:
                return avg_B/(avg_A+avg_B) 
            return avg_A/(avg_A+avg_B) 
        else:
            counter_A_better = 0
            rewards_rand = self.get_rand_indexes(self.kappa)
            for index  in rewards_rand:
                if  self.memory[index][A] > self.memory[index][B]:
                    counter_A_better += 1
            return counter_A_better / len(rewards_rand)
        
    
    def choose(self):
        # TODO first trial
        p_surprise = self.surprise()
        p_inertia = 0
        coeff_inertia = (1-p_surprise)*(1-self.eta)
        if self.memory[-1][0] == A:
            p_inertia = coeff_inertia
        coeff_new_dec = 1 - coeff_inertia
        p_old = self.delta**((len(self.memory)-1)/len(self.memory))
        p_new = 1 - p_old
        p_sample = self.get_sample_prob()
        p_desc = self.padescriptor
        p_new_desiciion = coeff_new_dec*(p_old*p_desc + p_new*p_sample)
        total_prob = p_inertia + p_new_desiciion
        # TODO change to flipping a coin
        if total_prob > 0.5:
            return A
        else:
            return B


    def update(self, choice, a_payoff, b_payoff):
        self.memory.append((choice, a_payoff, b_payoff))




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