import os
import pathlib
import tempfile
from typing import Union, Dict, Optional

import mlflow
import torch 

from .AbstractCallback import AbstractCallback

class MlflowLogger(AbstractCallback):
    """
    Callback to log metrics to MLflow.
    """

    def __init__(
            self, 
            name: str,
            mlflow_uri: Union[pathlib.Path, str] = None,
            mlflow_experiment_name: Optional[str] = None,
            mlflow_start_run_args: dict = None,
            mlflow_log_params_args: dict = None,
            mlflow_tags: dict = None,
            _log_best_model: bool = True,
            _best_model_artifact_name: str = 'weights_best.pth',
            _log_epoch_model: bool = True,
            _temp_dir: Optional[str] = None
        ):
        """
        Initialize the MlflowLogger callback.

        :param name: Name of the callback.
        :param mlflow_uri: URI for the MLflow tracking server, defaults to None.
        If a path is specified, the logger class will call set_tracking_uri to that supplied path 
        thereby initiating a new tracking server. 
        If None (default), the logger class will not tamper with mlflow server to enable logging to a global server
        initialized outside of this class. 
        :type mlflow_uri: pathlib.Path or str, optional
        :param mlflow_experiment_name: Name of the MLflow experiment, defaults to None, which will not call the 
        set_experiment method of mlflow and will use whichever experiment name that is globally configured. If a 
        name is provided, the logger class will call set_experiment to that supplied name.
        :type mlflow_experiment_name: str, optional
        :param mlflow_start_run_args: Additional arguments for starting an MLflow run, defaults to None.
        :type mlflow_start_run_args: dict, optional
        :param mlflow_log_params_args: Additional arguments for logging parameters to MLflow, defaults to None.
        :type mlflow_log_params_args: dict, optional
        :param _log_best_model: Whether to log the best model, defaults to True.
        :type _log_best_model: bool, optional
        :param _best_model_artifact_name: Name of the artifact for the best model, defaults to 'weights_best.pth'.
        :type _best_model_artifact_name: str, optional
        :param _log_epoch_model: Whether to log the model at the end of each epoch, defaults to True.
        :type _log_epoch_model: bool, optional
        :param _temp_dir: Temporary directory for saving model artifacts, defaults to None.
        If None, a temporary directory will be created using tempfile.TemporaryDirectory.
        :type _temp_dir: str, optional
        :param mlflow_tags: Tags to log with the MLflow run, defaults to None.
        :type mlflow_tags: dict, optional
        """
        super().__init__(name)

        if mlflow_uri is not None:
            try:
                mlflow.set_tracking_uri(mlflow_uri)
            except Exception as e:
                raise RuntimeError(f"Error setting MLflow tracking URI: {e}")                
        
        if mlflow_experiment_name is not None:
            try:
                mlflow.set_experiment(mlflow_experiment_name)
            except Exception as e:
                raise RuntimeError(f"Error setting MLflow experiment: {e}")

        self._mlflow_start_run_args = mlflow_start_run_args
        self._mlflow_log_params_args = mlflow_log_params_args
        self._mlflow_tags = mlflow_tags
        self._log_best_model = _log_best_model
        self._best_artifact_name = _best_model_artifact_name
        self._log_epoch_model = _log_epoch_model
        self._temp_dir = _temp_dir

    def on_train_start(self):
        """
        Called at the start of training.

        Calls mlflow start run and logs params if provided
        """

        if self._mlflow_start_run_args is None:
            pass
        elif isinstance(self._mlflow_start_run_args, Dict):
            mlflow.start_run(
                **self._mlflow_start_run_args
            )
        else:
            raise TypeError("mlflow_start_run_args must be None or a dictionary.")
        
        if self._mlflow_log_params_args is None:
            pass
        elif isinstance(self._mlflow_log_params_args, Dict):
            mlflow.log_params(
                self._mlflow_log_params_args
            )
        else:
            raise TypeError("mlflow_log_params_args must be None or a dictionary.")
        
        if self._mlflow_tags is not None:
            if not isinstance(self._mlflow_tags, dict):
                raise TypeError("mlflow_tags must be a dictionary.")
            for key, value in self._mlflow_tags.items():
                mlflow.set_tag(key, value)

    def on_epoch_end(self):
        """
        Called at the end of each epoch.

        Iterate over the most recent log items in trainer and call mlflow log metric
        """
        for key, values in self.trainer.log.items():
            if values is not None and len(values) > 0: 
                value = values[-1]
            else:
                value = None
            mlflow.log_metric(key, value, step=self.trainer.epoch)

        self.__log_model_artifacts()

    def on_train_end(self):
        """
        Called at the end of training.

        Saves trainer best model to a temporary directory and calls mlflow log artifact
        Then ends run
        """

        self.__log_model_artifacts()
        # # Save best model weights to a temporary directory and log artifacts
        # with tempfile.TemporaryDirectory(dir=self._temp_dir) as tmpdirname:
        #     weights_path = os.path.join(tmpdirname, self._artifact_name)
        #     torch.save(self.trainer.best_model, weights_path)
        #     mlflow.log_artifact(weights_path, artifact_path="models")

        # # also save the highest epoch model weights
        # with tempfile.TemporaryDirectory(dir=self._temp_dir) as tmpdirname:
        #     weights_path = os.path.join(tmpdirname, f'weights_{self.trainer.epoch}.pth')
        #     torch.save(self.trainer.model.state_dict(), weights_path)
        #     mlflow.log_artifact(weights_path, artifact_path="models")

        mlflow.end_run()
        
    def __log_model_artifacts(self):
        """
        Helper method for logging model artifacts to MLflow.
        """

        if self._log_best_model:
            with tempfile.TemporaryDirectory(dir=self._temp_dir) as tmpdirname:
                weights_path = os.path.join(tmpdirname, self._best_artifact_name)
                torch.save(self.trainer.best_model, weights_path)
                mlflow.log_artifact(weights_path, artifact_path="models")

        if self._log_epoch_model:
            # also save the highest epoch model weights
            with tempfile.TemporaryDirectory(dir=self._temp_dir) as tmpdirname:
                weights_path = os.path.join(tmpdirname, f'weights_{self.trainer.epoch}.pth')
                torch.save(self.trainer.model.state_dict(), weights_path)
                mlflow.log_artifact(weights_path, artifact_path="models")
