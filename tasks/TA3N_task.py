from abc import ABC
import torch
from utils import utils
from functools import reduce
import wandb
import tasks
from utils.logger import logger

from typing import Dict, Tuple


class TA3N_task(tasks.Task, ABC):
    """Ta3N."""
    
    def __init__(self, name: str, task_models: Dict[str, torch.nn.Module], batch_size: int, 
                 total_batch: int, models_dir: str, num_classes: int,
                 num_clips: int, model_args: Dict[str, float], args, **kwargs) -> None:
        """Create an instance of the action recognition model.

        Parameters
        ----------
        name : str
            name of the task e.g. action_classifier, domain_classifier...
        task_models : Dict[str, torch.nn.Module]
            torch models, one for each different modality adopted by the task
        batch_size : int
            actual batch size in the forward
        total_batch : int
            batch size simulated via gradient accumulation
        models_dir : str
            directory where the models are stored when saved
        num_classes : int
            number of labels in the classification task
        num_clips : int
            number of clips
        model_args : Dict[str, float]
            model-specific arguments
        """
        super().__init__(name, task_models, batch_size, total_batch, models_dir, args, **kwargs)
        self.model_args = model_args

        # self.accuracy and self.loss track the evolution of the accuracy and the training loss
        self.accuracy = utils.Accuracy(topk=(1, 5), classes=num_classes)
        self.accuracy_class = utils.Accuracy(topk=(1, 5), classes=num_classes)
        self.accuracy_td = utils.Accuracy(classes=2)
        self.accuracy_sd = utils.Accuracy(classes=2)
        
        self.loss = utils.AverageMeter()
        self.loss_class = utils.AverageMeter() ########## controlla ##########
        self.loss_td = utils.AverageMeter()
        self.loss_sd = utils.AverageMeter()
        
        self.num_clips = num_clips

        # Use the cross entropy loss as the default criterion for the classification task
        self.criterion_class = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        self.criterion_td = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        self.criterion_sd = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        # Initializeq the model parameters and the optimizer
        optim_params = {}
        self.optimizer = dict()
        for m in self.modalities:
            optim_params[m] = filter(lambda parameter: parameter.requires_grad, self.task_models[m].parameters())
            self.optimizer[m] = torch.optim.SGD(optim_params[m], model_args[m].lr,
                                                weight_decay=model_args[m].weight_decay,
                                                momentum=model_args[m].sgd_momentum)

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward step of the task

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            a dictionary that stores the input data for each modality 

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            output logits and features
        """
        # logits_class = {}
        # logits_td = {}
        # logits_sd = {}
        logits = {"class":{},
                  "td":{},
                  "sd":{}}
        ""

        features = {}
        for i_m, m in enumerate(self.modalities):
            logits["sd"][m],logits["td"][m],logits["class"][m]= self.task_models[m](x=data[m], **kwargs)

        return logits

    def compute_loss(self, logits, label_class: torch.Tensor,label_d: torch.Tensor, loss_weight: float=1.0,domain = "source"):
        """Fuse the logits from different modalities and compute the classification loss.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        loss_weight : float, optional
            weight of the classification loss, by default 1.0
        """
        if(domain == "source"): #source
            fused_logits_class = reduce(lambda x, y: x + y, logits["class"].values())
            fused_logits_sd = reduce(lambda x, y: x + y, logits["sd"].values())
            fused_logits_td = reduce(lambda x, y: x + y, logits["td"].values())
        
            self.loss_class.update(self.criterion_class(fused_logits_class, label_class))
            self.loss_sd.update(self.criterion_sd(fused_logits_sd, label_d))
            self.loss_td.update(self.criterion_td(fused_logits_td, label_d))

            loss = self.loss_class.val+self.loss_sd.val+self.loss_td.val
            # Update the loss value, weighting it by the ratio of the batch size to the total 
            # batch size (for gradient accumulation)
            self.loss.update(torch.mean(loss_weight * loss) / (self.total_batch / self.batch_size), self.batch_size)
        else: #target
            fused_logits_sd = reduce(lambda x, y: x + y, logits["sd"].values())
            fused_logits_td = reduce(lambda x, y: x + y, logits["td"].values())
            
            
            self.loss_sd.add(self.criterion_sd(fused_logits_sd, label_d))
            self.loss_td.add(self.criterion_td(fused_logits_td, label_d))

            loss = self.loss_sd.val+self.loss_td.val
            self.loss.add(torch.mean(loss_weight * loss) / (self.total_batch / self.batch_size), self.batch_size)

    
    def compute_accuracy(self, logits_source: Dict[str, torch.Tensor], label: torch.Tensor):
        """Fuse the logits from different modalities and compute the classification accuracy.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        """
        logits = logits_source["class"]
        fused_logits = reduce(lambda x, y: x + y, logits.values())
        self.accuracy.update(fused_logits, label)

    def wandb_log(self):
        """Log the current loss and top1/top5 accuracies to wandb."""
        logs = {
            'loss verb': self.loss.val, 
            'top1-accuracy': self.accuracy.avg[1],
            'top5-accuracy': self.accuracy.avg[5]
        }

        # Log the learning rate, separately for each modality.
        for m in self.modalities:
            logs[f'lr_{m}'] = self.optimizer[m].param_groups[-1]['lr']
        wandb.log(logs)

    def reduce_learning_rate(self):
        """Perform a learning rate step."""
        for m in self.modalities:
            prev_lr = self.optimizer[m].param_groups[-1]["lr"]
            new_lr = self.optimizer[m].param_groups[-1]["lr"] / 10
            self.optimizer[m].param_groups[-1]["lr"] = new_lr

            logger.info(f"Reducing learning rate modality {m}: {prev_lr} --> {new_lr}")

    def reset_loss(self):
        """Reset the classification loss.

        This method must be called after each optimization step.
        """
        self.loss_class.reset()
        self.loss_sd.reset()
        self.loss_td.reset()

        self.loss.reset()

    def reset_acc(self):
        """Reset the classification accuracy."""
        self.accuracy.reset()

    def step(self):
        """Perform an optimization step.

        This method performs an optimization step and resets both the loss
        and the accuracy.
        """
        super().step()
        self.reset_loss()
        self.reset_acc()

    def backward(self, retain_graph: bool = False):
        """Compute the gradients for the current value of the classification loss.

        Set retain_graph to true if you need to backpropagate multiple times over
        the same computational graph.

        Parameters
        ----------
        retain_graph : bool, optional
            whether the computational graph should be retained, by default False
        """
        self.loss.val.backward(retain_graph=retain_graph)

    def get_losses(self):
        losses = torch.zeros([3])
        losses[0] = torch.mean(self.loss_class.val)
        losses[1] = torch.mean(self.loss_sd.val)
        losses[2] = torch.mean(self.loss_td.val)
        return losses