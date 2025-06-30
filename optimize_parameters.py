from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from simulate_and_evaluate_model import evaluate_against_dataset

# Define the parameter search space
search_space = [
    Integer(4,9, name='kappa'),
    Real(0.1, 0.9, name='delta'),
    Real(0.6, 0.9, name='eta'),
    Categorical(['uniform', 'half_gaussian_bell'], name='distribution'),
    Integer(1, 3, name='radius'),
    Integer(1, 5, name='scale'),
]

file_path = "Training 2025.04.22DM.xlsx"  # Adjust path if needed

def objective(params):
    kappa, delta, eta, distribution, radius, scale = params
    # Call the loss function with all parameters
    return evaluate_against_dataset(
        file_path,
        kappa=int(kappa),
        delta=delta,
        eta=eta,
        distribution=distribution,
        radius=radius,
        scale=scale
    )

if __name__ == "__main__":
    res = gp_minimize(
        objective,
        search_space,
        n_calls=30,
        random_state=0
    )
    print("Best loss:", res.fun)
    print("Best parameters:")
    print(f"kappa={res.x[0]}, delta={res.x[1]}, eta={res.x[2]}, distribution={res.x[3]}, radius={res.x[4]}, scale={res.x[5]}") 