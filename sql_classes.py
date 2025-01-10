# This module defines the SQLAlchemy ORM classes for database interaction.
# It includes the schema for storing experiments, metrics, and model states.

from sqlalchemy import (
    Column, Integer, Float, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import LargeBinary

Base = declarative_base()

class ExperimentResult(Base):
    __tablename__ = 'experiment_results'

    id = Column(Integer, primary_key=True)
    ind_trajectory = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)

    # New column to store the serialized model state
    model_state = Column(LargeBinary, nullable=True)

    # Relationship to ExperimentMetrics
    metrics = relationship("ExperimentMetrics", back_populates="experiment", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint('learning_rate', 'ind_trajectory', name='_lr_ind_uc'),
    )

class ExperimentMetrics(Base):
    __tablename__ = 'experiment_metrics'

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiment_results.id'), nullable=False)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=True)
    test_loss  = Column(Float, nullable=True)
    train_acc  = Column(Float, nullable=True)
    test_acc   = Column(Float, nullable=True)

    # Relationship back to ExperimentResult
    experiment = relationship("ExperimentResult", back_populates="metrics")

    __table_args__ = (
        UniqueConstraint('experiment_id', 'epoch', name='_experiment_epoch_uc'),
    )