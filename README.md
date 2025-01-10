
# SQLAlchemy for Experiment Tracking

This project demonstrates how to use **SQLAlchemy** for managing machine learning experiment tracking, 
including storing metadata, model states, and metrics in a relational database. It integrates seamlessly 
with **PyTorch** for training and evaluating models and supports **multiprocessing** for parallel execution.

---

## Features

- Tracks experiment metadata (e.g., learning rates, epochs, trajectories).
- Saves and resumes model states and optimizer states (supports optimizers like Adam and SGD).
- Uses multiprocessing for running multiple experiments in parallel.
- Generates detailed metrics, plots, and analytics for experiments.
- Fully scalable with database-backed experiment management using SQLAlchemy.

---

## Requirements

To run this project, you’ll need the following installed:

- Python 3.8+
- PyTorch
- SQLAlchemy
- scikit-learn
- Matplotlib
- numpy
- pandas

---

## File Structure

```
.
├── experiment_running.py       # Script to manage and track experiments
├── model_functions.py          # Reusable PyTorch functions for training and evaluation
├── sql_classes.py              # SQLAlchemy classes for database schema
├── Test_SQLAlchemy.ipynb       # Jupyter notebook for testing and visualizing SQLAlchemy usage
└── README.md                   # Project documentation
```

---

## Getting Started

### 1. Initialize the Database
Ensure that the database is set up before running any experiments. The database schema is defined in `sql_classes.py`. 
The database will be created automatically when you run an experiment.

### 2. Running Experiments
To run experiments, use the `experiment_running.py` script. You can define parameters directly or pass them 
as command-line arguments.

---

## Visualizing Results

After running experiments, use the metrics stored in the database to generate plots. 
You can load the data using SQLAlchemy and analyze or visualize it with tools like Matplotlib or pandas.

The Jupyter notebook `Test_SQLAlchemy.ipynb` includes examples of loading data and generating plots.

---

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/)
