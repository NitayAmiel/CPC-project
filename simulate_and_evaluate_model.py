import numpy as np
import pandas as pd
import random
import math

A = 1
B = 2

def sample_from_list(lst, n, distribution="half_gaussian_bell", spread=3):
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
    def __init__(self, delta=0.5, kappa=10, eta=0.1, padescriptor=None, distribution = "uniform", forgone=False):
        self.delta = delta
        self.kappa = kappa
        self.eta = eta
        self.padescriptor = padescriptor
        self.distribution = distribution
        self.memory = []  # Each entry will be (choice, A_payoff, B_payoff) where choice is 1 for A, 0 for B
        self.forgone = forgone

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
        if mean_reward < V_j_t:
            return 0.0
        return abs(mean_reward - V_j_t) / (max_reward - min_reward)
    

    def get_rand_indexes(self, num_of_idx, type = None):
        idx_list = []
        for idx in range(len(self.memory)):
            if type == None or self.memory[idx][0] == type:
                idx_list.append(idx)
        if len(idx_list) == 0:
            return []
        
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
            
            avg_A = np.mean(reward_rand_A) if len(reward_rand_A) > 0 else 0
            avg_B = np.mean(reward_rand_B) if len(reward_rand_B) > 0 else 0

            total_competitions = len(rewards_rand_index_A) + len(rewards_rand_index_B)
            
            # print(f"kappa: {self.kappa}, kappa_A: {kappa_A}, kappa_B: {kappa_B}, rewards_rand_index_A: {rewards_rand_index_A}, rewards_rand_index_B: {rewards_rand_index_B}")
            # exit(0)
            # assert (total_competitions == len(rewards_rand_index_A ) + len(rewards_rand_index_B))

            a_won = 0
            equal_competitions = 0
            for i in range(len(self.memory)):
                if i in rewards_rand_index_A:
                    if(self.memory[i][A] > avg_B):
                        a_won += 1
                    elif self.memory[i][A] == avg_B:
                        equal_competitions += 1
                elif i in rewards_rand_index_B:
                    if(self.memory[i][B] < avg_A):
                        a_won += 1
                    elif self.memory[i][B] == avg_A:
                        equal_competitions += 1

            return (a_won + 0.5*equal_competitions) / total_competitions


        else:
            counter_A_better = 0
            counter_equals = 0
            rewards_rand = self.get_rand_indexes(self.kappa, type=None)
            for index  in rewards_rand:
                if  self.memory[index][A] > self.memory[index][B]:
                    counter_A_better += 1
                elif self.memory[index][A] == self.memory[index][B]:
                    counter_equals += 1
            return (counter_A_better + 0.5*counter_equals) / len(rewards_rand)
        
    def get_last_choice(self):
        if len(self.memory) == 1:
            return None
        return self.memory[-2][0]
    
    def get_last_A_B_choices(self):
        last_A_idx = -1
        last_B_idx = -1
        for idx in range(len(self.memory)-1, -1, -1):
            if last_A_idx == -1 and self.memory[idx][A] != None:
                last_A_idx = idx
            if last_B_idx == -1 and self.memory[idx][B] != None:
                last_B_idx = idx
            if last_A_idx != -1 and last_B_idx != -1:
                break
        last_A_reward = self.memory[last_A_idx][A] if last_A_idx != -1 else None
        last_B_reward = self.memory[last_B_idx][B] if last_B_idx != -1 else None
        return last_A_reward, last_B_reward
    
    def choose(self):
        if len(self.memory) == 0 :
            return A if random.random() < self.padescriptor else B
        if len(self.memory) == 1 and self.forgone: #check later what to do with this
            return A if self.memory[0][0] == B else B
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
        # breakpoint()
        return A if random.random() < total_prob else B

    def update(self, choice, a_payoff, b_payoff):
        self.memory.append((choice, a_payoff, b_payoff))




def simulate_task(p_a_descriptor, a_prob, a1, a2, b_prob, b1, b2, forgone, n_participants, n_trials, corr_ab, kappa, delta, eta ,distribution , radius=1 ):
    results = {
        'arate1': 0.0, 'arate2': 0.0, 'arate3': 0.0, 'arate4': 0.0,
        'afoka': 0.0, 'afrega': 0.0, 'afregb': 0.0, 'afokb': 0.0
    }
    results_custom = { 'total_a_after_ok_a' : 0 , 'total_a_after_bad_a' : 0, 'total_a_after_bad_b' : 0
                      , 'total_a_after_ok_b' : 0 , 'total_A' : [0,0,0,0]}
    total_results_custom = { 'total_after_ok_a' : 0 , 'total_after_bad_a' : 0, 'total_after_bad_b' : 0
                      , 'total_after_ok_b' : 0 }
    def get_reward():
        randA_choice = random.random()
        randB_choice = randA_choice
        if corr_ab == -1:
            randB_choice = 1 - randA_choice
        elif corr_ab == 0:
            randB_choice = random.random()

        reward_A = a1 if randA_choice < a_prob else a2
        reward_B = b1 if randB_choice < b_prob else b2
        return ( 0 , reward_A, reward_B)
        

    for idx in range(n_participants):
        kappa_curr = kappa + (radius - (idx % (radius*2 + 1)))
        agent = ISamplerAgent(padescriptor=p_a_descriptor, forgone=forgone, kappa= kappa_curr, delta = delta, eta = eta, distribution=distribution)

        for t in range(n_trials):
            # breakpoint()
            # if t % 25 == 0:
                # breakpoint()
            block = t // (n_trials // 4)
            choice = agent.choose()
            rewards = get_reward()
            if choice == A:
                results_custom['total_A'][block] += 1
            if forgone:
                if choice == A:
                    agent.update(choice, rewards[A], None)
                else:
                    agent.update(choice,  None, rewards[B])
            else:
                agent.update(choice, rewards[A], rewards[B])

            last_choice = agent.get_last_choice()
            if last_choice == None:
                continue
            a_last_payoff, b_last_payoff = agent.get_last_A_B_choices()
            if a_last_payoff is None or b_last_payoff is None:
                continue
            add = 0
            if choice == A:
                add = 1
            if last_choice == A:
                if a_last_payoff > b_last_payoff:
                    results_custom['total_a_after_ok_a'] += add
                    total_results_custom['total_after_ok_a'] += 1
                else:
                    results_custom['total_a_after_bad_a'] += add
                    total_results_custom['total_after_bad_a'] += 1
            else:
                if a_last_payoff > b_last_payoff:
                    results_custom['total_a_after_bad_b'] += add
                    total_results_custom['total_after_bad_b'] += 1
                else:
                    results_custom['total_a_after_ok_b'] += add
                    total_results_custom['total_after_ok_b'] += 1

    results['arate1'] = results_custom['total_A'][0] / (n_participants*(n_trials/4))
    results['arate2'] = results_custom['total_A'][1] / (n_participants*(n_trials/4))
    results['arate3'] = results_custom['total_A'][2] / (n_participants*(n_trials/4))
    results['arate4'] = results_custom['total_A'][3] / (n_participants*(n_trials/4))
    
    results['afoka'] = results_custom['total_a_after_ok_a'] / total_results_custom['total_after_ok_a'] if total_results_custom['total_after_ok_a'] != 0 else 0
    results['afrega'] = results_custom['total_a_after_bad_a'] / total_results_custom['total_after_bad_a'] if total_results_custom['total_after_bad_a'] != 0 else 0
    results['afregb'] = results_custom['total_a_after_bad_b'] / total_results_custom['total_after_bad_b'] if total_results_custom['total_after_bad_b'] != 0 else 0
    results['afokb'] = results_custom['total_a_after_ok_b'] / total_results_custom['total_after_ok_b'] if total_results_custom['total_after_ok_b'] != 0 else 0

    return results


def evaluate_against_dataset(excel_path, kappa, delta, eta, distribution, radius, scale):
    df = pd.read_excel(excel_path)
    target_cols = ['arate1', 'arate2', 'arate3', 'arate4', 'afoka', 'afrega', 'afregb', 'afokb']
    df_clean = df[df['ChatGPTo'].notna()]
    
    # Filter to only include rows where forgone=0
    df_clean = df_clean[df_clean['forgone'] == 1]
    
    for col in target_cols:
        df_clean = df_clean[df_clean[col].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
    df_clean[target_cols] = df_clean[target_cols].astype(float)

    predicted_results = []

    for _, row in df_clean.iterrows():
        sim_result = simulate_task(
            p_a_descriptor=float(row['ChatGPTo']),
            a_prob=float(row['pa1']), a1=float(row['a1']), a2=float(row['a2']),
            b_prob=float(row['pb1']), b1=float(row['b1']), b2=float(row['b2']),
            forgone=bool(1 - row['forgone']), n_participants=scale*int(row['n']), n_trials=100, corr_ab=int(row['corrAB'])
        , kappa = kappa, delta = delta, eta = eta, distribution=distribution,  radius = radius)
        predicted_results.append(sim_result)

    pred_df = pd.DataFrame(predicted_results)
    gt_df = df_clean[target_cols].reset_index(drop=True)
    comparison_df = pd.concat([pred_df.add_prefix('pred_'), gt_df.add_prefix('gt_')], axis=1)
    comparison_df['MSE'] = ((pred_df.round(3) - gt_df.round(3)) ** 2).mean(axis=1)
    print("\n--- Summary ---")
    print(comparison_df.mean())
    print("\n--- Detailed Comparison ---")
    print(comparison_df.head())
    return comparison_df['MSE'].mean()

    return comparison_df

if __name__ == "__main__":
    hyper_parameters = {
        "kappa": 9,
        "delta": 0.4143158063681155,
        "eta": 0.7492172409901976,
        "distribution": "uniform",
        "radius": 3,
        "scale": 6
    }
    file_path = "Training 2025.04.22DM.xlsx"  # Adjust path if needed
    
    # Use the hyper_parameters dictionary to call evaluate_against_dataset
    lossval =  evaluate_against_dataset(
        file_path,
        kappa=hyper_parameters["kappa"],
        delta=hyper_parameters["delta"],
        eta=hyper_parameters["eta"],
        distribution=hyper_parameters["distribution"],
        radius=hyper_parameters["radius"],
        scale=hyper_parameters["scale"]
    )
    print(lossval)
    
    exit(0)
    sim_result = simulate_task(
            p_a_descriptor=0.215,
            a_prob=1, a1=0, a2=0,
            b_prob=0.1, b1=10, b2=-1,
            forgone=True, n_participants=1, n_trials=100, corr_ab=0, kappa=4, delta=0.2, eta = 0.8
        )
    
    # print("\nSimulation Results:")
    # print("-----------------")
    # print("Block Averages:")
    # print(f"  Block 1: {sim_result['arate1']:.3f}")
    # print(f"  Block 2: {sim_result['arate2']:.3f}")
    # print(f"  Block 3: {sim_result['arate3']:.3f}")
    # print(f"  Block 4: {sim_result['arate4']:.3f}")
    # print("\nCondition Averages:")
    # print(f"  After OK A: {sim_result['afoka']:.3f}")
    # print(f"  After Regret A: {sim_result['afrega']:.3f}")
    # print(f"  After Regret B: {sim_result['afregb']:.3f}")
    # print(f"  After OK B: {sim_result['afokb']:.3f}")