# -*- coding: utf-8 -*-
# @Time   : 2022/8/2
# @Author : Ayush Thakur
# @Email  : ayusht@wandb.com

r"""
recbole.utils.wandblogger
로부터 수정했습니다.
"""
import numpy as np


class WandbLogger(object):
    """WandbLogger to log metrics to Weights and Biases."""

    def __init__(self, config):
        """
        Args:
            config (dict): A dictionary of parameters used by RecBole.
        """
        self.config = config
        self.log_wandb = config['wandb']
        self.setup()

    def setup(self):
        if self.log_wandb:
            try:
                import wandb

                self._wandb = wandb
            except ImportError:
                raise ImportError(
                    "To use the Weights and Biases Logger please install wandb."
                    "Run `pip install wandb` to install it."
                )

            # Initialize a W&B run
            if self._wandb.run is None:
                self._wandb.init(project=self.config.wandb_project, name=self.config.wandb_experiment_name, 
                                notes=self.config.memo if hasattr(self.config, 'memo') else None,
                                tags=self.config.model, 
                                )
            self._set_steps()

    def log_metrics(self, metrics, head="train", epoch = None, commit=True):
        if self.log_wandb:
            if head:
                metrics = self._add_head_to_metrics(metrics, head)
            
            if epoch is not None:
                metrics['epoch'] = epoch  
            
            self._wandb.log(metrics, commit=commit)

    def log_eval_metrics(self, metrics, head="eval"):
        if self.log_wandb:
            metrics = self._add_head_to_metrics(metrics, head)
            for k, v in metrics.items():
                self._wandb.run.summary[k] = v

    def _set_steps(self):
        self._wandb.define_metric("train/*", step_metric="epoch")
        self._wandb.define_metric("valid/*", step_metric="epoch")

    def _add_head_to_metrics(self, metrics, head):
        head_metrics = dict()
        for k, v in metrics.items():
            if isinstance(v, (np.ndarray, list)):
                for i, value in enumerate(v):
                    topk = self.config.topks[i] 
                    head_metrics[f"{head}/{k}_at_{topk}"] = value
            else :
                topk = self.config.topks[0] 
                head_metrics[f"{head}/{k}_at_{topk}"] = v 

        return head_metrics
    
    