# This script manages the execution and tracking of machine learning experiments.
# It utilizes the SQLAlchemy ORM to store experiment metadata, results, and model states.

import io

import torch.optim as optim
from torch.utils.data import DataLoader

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from model_functions import *
from sql_classes import *

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def get_session(db_path="sqlite:///experiments.db"):
    engine = create_engine(db_path, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)
    return Session()

def save_state_to_bytes(model, optimizer):
    buffer = io.BytesIO()
    torch.save({'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()}, buffer)
    buffer.seek(0)
    return buffer.read()

def load_state_from_bytes(model, optimizer, byte_data):
    buffer = io.BytesIO(byte_data)
    buffer.seek(0)
    # Specify weights_only=True to suppress the warning and load only state_dicts
    checkpoint = torch.load(buffer, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

def worker(args):
    run_experiment(*args)

def run_experiment(
    learning_rate,
    target_epochs,
    ind_trajectory,
    batch_size=32,
    hidden_dim=16,
    db_path="sqlite:///experiments.db",
    optimizer_type="Adam"  # Add optimizer type as a parameter
):
    session = get_session(db_path)

    try:
        # Retrieve existing experiment or create a new one
        existing_experiment = (
            session.query(ExperimentResult)
            .filter_by(learning_rate=learning_rate, ind_trajectory=ind_trajectory)
            .first()
        )

        if existing_experiment is None:
            # Create a new experiment if none exists
            existing_experiment = ExperimentResult(
                learning_rate=learning_rate,
                ind_trajectory=ind_trajectory,
                model_state=None  # Initialize with no model state
            )
            session.add(existing_experiment)
            session.commit()
            start_epoch = 1
            print(f"Starting new experiment: LR={learning_rate}, Trajectory={ind_trajectory}")
        else:
            # Determine the starting epoch
            last_metric = (
                session.query(ExperimentMetrics)
                .filter_by(experiment_id=existing_experiment.id)
                .order_by(ExperimentMetrics.epoch.desc())
                .first()
            )
            start_epoch = last_metric.epoch + 1 if last_metric else 1
            print(f"Resuming experiment: LR={learning_rate}, Trajectory={ind_trajectory}, Start Epoch={start_epoch}")

        # Initialize model
        model = SimpleModel(input_dim=2, hidden_dim=hidden_dim, output_dim=2)

        # Initialize optimizer
        if optimizer_type == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Load model and optimizer state if available
        if existing_experiment.model_state:
            load_state_from_bytes(model, optimizer, existing_experiment.model_state)

        # Prepare datasets and dataloaders
        X_train, X_test, y_train, y_test, scaler = create_data(random_state = ind_trajectory)

        train_dataset = MoonDataset(X_train, y_train)
        test_dataset = MoonDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        # Limit the number of threads in PyTorch explicitly
        torch.set_num_threads(1)

        # Training loop
        for epoch in range(start_epoch, target_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader)
            test_loss, test_acc = evaluate(model, criterion, test_loader)

            print(
                f"[LR={learning_rate}, Trajectory={ind_trajectory}, Epoch={epoch}/{target_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            )

            # Save metrics for the current epoch
            epoch_metrics = ExperimentMetrics(
                experiment_id=existing_experiment.id,
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                train_acc=train_acc,
                test_acc=test_acc,
            )
            session.add(epoch_metrics)
            session.commit()

            # Save the current model and optimizer state
            existing_experiment.model_state = save_state_to_bytes(model, optimizer)
            session.commit()

    except Exception as e:
        print(f"Error in experiment (LR={learning_rate}, Trajectory={ind_trajectory}): {e}")
        session.rollback()  # Roll back the transaction in case of an error
    finally:
        session.close()