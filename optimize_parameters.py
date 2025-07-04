from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from simulate_and_evaluate_model import evaluate_against_dataset

# Define the parameter search space
search_space = [
    Integer(2,9, name='kappa'),
    Real(0.0, 0.9, name='delta'),
    Real(0.0, 0.9, name='eta'),
    # Categorical(['half_gaussian_bell', 'uniform'], name='distribution'),
    Integer(0, 1, name='radius'),
    Integer(1, 20, name='scale'),
    Real(0.0, 1.0, name='alpha'),
    Real(0.0, 1.0, name='beta'),
    Real(0.0, 3.0, name='lambda_'),
    Real(0.1, 0.3, name='gamma'),
]

file_path = "Training 2025.04.22DM.xlsx"  # Adjust path if needed

def objective(params):
    kappa, delta, eta, radius, scale, alpha, beta, lambda_, gamma = params
    # Call the loss function with all parameters
    return evaluate_against_dataset(
        file_path,
        kappa=int(kappa),
        delta=delta,
        eta=eta,
        distribution="uniform",
        radius=radius,
        scale=scale,
        alpha=alpha,
        beta=beta,
        lambda_=lambda_,
        gamma=gamma
    )

if __name__ == "__main__":
    res = gp_minimize(
        objective,
        search_space,
        n_calls=100,
        random_state=0
    )
    print("Best loss:", res.fun)
    print("Best parameters:")
    print(f"kappa={res.x[0]}, delta={res.x[1]}, eta={res.x[2]}, radius={res.x[3]}, scale={res.x[4]}, alpha={res.x[5]}, beta={res.x[6]}, lambda_={res.x[7]}, gamma={res.x[8]}") 