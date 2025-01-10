# This script runs multiple machine learning experiments in parallel using multiprocessing.
# It coordinates with the database to store results and ensures experiments resume properly.

from experiment_running import *
import argparse
import numpy as np

def parallel_experiments(
    learning_rates,
    n_trajectories,
    target_epochs,
    db_path="sqlite:///experiments.db",
    num_processes=2
):
    X_train, X_test, y_train, y_test, scaler = create_data()

    tasks = []
    for lr in learning_rates:
        for ind_traj in range(n_trajectories):
            tasks.append((
                lr, target_epochs, ind_traj,
                X_train, y_train, X_test, y_test,
                32, 16, db_path
            ))

    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(worker, tasks)
    else:
        # Run sequentially for debugging
        for t in tasks:
            worker(t)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parallel Experiments with PyTorch and SQLAlchemy")

    parser.add_argument(
        '--learning_rates',
        type=float,
        nargs='+',
        default=np.logspace(-3, 0, 10).tolist(),
        help='List of learning rates (e.g., --learning_rates 0.001 0.01 0.1 1.0)'
    )

    parser.add_argument(
        '--n_trajectories',
        type=int,
        default=10,
        help='Number of trajectories (e.g., --n_trajectories 10)'
    )

    parser.add_argument(
        '--target_epochs',
        type=int,
        default=10,
        help='Number of target epochs (e.g., --target_epochs 10)'
    )

    parser.add_argument(
        '--num_processes',
        type=int,
        default=2,
        help='Number of parallel processes (e.g., --num_processes 2)'
    )

    parser.add_argument(
        '--db_path',
        type=str,
        default="sqlite:///experiments.db",
        help='Database path (e.g., --db_path "sqlite:///experiments.db")'
    )

    return parser.parse_args()

if __name__ == "__main__":
    # 1. Set the start method to 'spawn'
    multiprocessing.set_start_method("spawn")

    # 2. Parse command-line arguments
    args = parse_arguments()

    learning_rates = args.learning_rates
    n_trajectories = args.n_trajectories
    target_epochs = args.target_epochs
    num_processes = args.num_processes
    db_path = args.db_path

    # 3. Create the database tables **once** in the main process
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)

    # 4. Run experiments in parallel
    parallel_experiments(
        learning_rates,
        n_trajectories,
        target_epochs,
        db_path=db_path,
        num_processes=num_processes  # Adjust based on your CPU cores
    )

    # Example of running with higher target_epochs to resume:
    # python parallel_experiments.py --target_epochs 10

