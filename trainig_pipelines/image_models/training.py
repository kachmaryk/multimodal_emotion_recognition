import os
import sys
import time
import wandb
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Add the ml_part/models_tryouts/second_model/early-stopping-pytorch directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from early_stopping import EarlyStopping


class Trainer:
    def __init__(self, net, optimizer, epochs,
                 scheduler_type, scheduler_step_size=220,
                 cycle_momentum=False, patience=45,
                 use_cuda=True, gpu_num=0,
                 checkpoint_folder="./checkpoints",
                 num_classes=8, is_multilabel=False):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.patience = patience
        self.num_classes = num_classes
        self.is_multilabel = is_multilabel
        self.scheduler_step_size = scheduler_step_size
        self.cycle_momentum = cycle_momentum

        if self.scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10,
                                               min_lr=0.00001, verbose=True)
        elif self.scheduler_type == 'CyclicLR':
            self.scheduler = CyclicLR(self.optimizer,
                                      base_lr=0.00001,
                                      max_lr=0.001,
                                      step_size_up=self.scheduler_step_size,
                                      step_size_down=self.scheduler_step_size,
                                      mode='triangular',
                                      cycle_momentum=self.cycle_momentum)
        self.emotions = [
            "Curiosity", "Uncertainty", "Excitement", "Happiness",
            "Surprise", "Disgust", "Fear", "Frustration"
        ]

        self.epochs = epochs
        self.use_cuda = use_cuda
        self.gpu_num: int = gpu_num

        self.checkpoints_folder = checkpoint_folder
        self.checkpoint_path = os.path.join(
            self.checkpoints_folder,
            f"{str(dt.date.today()).replace('-', '_')}_major_emotion"
        )

        self.criterion = self.get_loss_function(is_multilabel=is_multilabel)

        if self.use_cuda:
            print(f"Running on GPU?", self.use_cuda, "- gpu_num: ", self.gpu_num)
            self.net = net.cuda('cuda:%i' % self.gpu_num)

            print("torch.cuda.memory_allocated: %.2f MB" % (torch.cuda.memory_allocated(self.gpu_num) / 1024 ** 2))
            print("torch.cuda.max_memory_allocated: %.2f MB" % (
                        torch.cuda.max_memory_allocated(self.gpu_num) / 1024 ** 2))
            print("torch.cuda.memory_reserved: %.2f MB" % (torch.cuda.memory_reserved(self.gpu_num) / 1024 ** 2))
            print("torch.cuda.max_memory_reserved: %.2f MB" % (
                    torch.cuda.max_memory_reserved(self.gpu_num) / 1024 ** 2))

            free, total = torch.cuda.mem_get_info(self.gpu_num)
            print(f"free GPU memory = {free / (1024 ** 2):.2f} MB")
            print(f"total GPU memory = {total / (1024 ** 2):.2f} MB")

            print(torch.cuda.memory_summary())
        else:
            self.net = net

    def train_model(self, train_loader, validation_loader, test_loader):
        # Update checkpoint folder name with wandb's run_name
        run_name = wandb.run.name
        self.checkpoint_path = f'{self.checkpoint_path}_{run_name}'
        os.makedirs(self.checkpoint_path, exist_ok=False)

        images_path = os.path.join(self.checkpoint_path, 'eval_images')
        os.makedirs(images_path, exist_ok=False)

        log_file = open(os.path.join(self.checkpoint_path, 'logs_output.txt'), 'w')

        # Initialize the early_stopping objects
        """ Losses checkpoints """
        early_stopping_train_loss = EarlyStopping(
            patience=None,
            path=os.path.join(self.checkpoint_path,  "best_checkpoint_train_loss.pt"),
            is_loss=True
        )
        early_stopping_val_loss = EarlyStopping(
            patience=None,
            path=os.path.join(self.checkpoint_path,  "best_checkpoint_val_loss.pt"),
            is_loss=True
        )

        """ Classification checkpoints"""
        early_stopping_val_acc = EarlyStopping(
            patience=None,
            path=os.path.join(self.checkpoint_path,  "best_checkpoint_val_acc.pt")
        )
        early_stopping_val_mAP = EarlyStopping(
            patience=None,
            path=os.path.join(self.checkpoint_path,  "best_checkpoint_val_mAP.pt")
        )
        early_stopping_val_MAR = EarlyStopping(
            patience=self.patience,
            path=os.path.join(self.checkpoint_path,  "best_checkpoint_val_MAR.pt")
        )

        early_stopping_test_MAR = EarlyStopping(
            patience=None,
            path=os.path.join(self.checkpoint_path,  "best_checkpoint_test_MAR.pt")
        )

        # Training loop
        for epoch in range(self.epochs):
            print("\n", f"Start of epoch {epoch}",
                  "torch.cuda.memory_allocated: %.2f MB" % (torch.cuda.memory_allocated(self.gpu_num) / 1024 ** 2))

            start_time = time.time()

            self.net.train()
            running_train_loss = 0.0

            y_pred_train = torch.empty(0)
            y_true_train = torch.empty(0)
            for train_iter, data in enumerate(tqdm(train_loader), 0):
                inputs, labels, *metadata = data

                if self.use_cuda:
                    inputs, labels = inputs.cuda('cuda:%i' % self.gpu_num), labels.cuda('cuda:%i' % self.gpu_num)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                if self.scheduler_type == "CyclicLR":
                    self.scheduler.step()

                running_train_loss += loss.item()

                # Classification
                predicted = torch.max(outputs, 1).indices
                y_pred_train = torch.cat((y_pred_train, predicted.view(predicted.shape[0]).detach().cpu()))
                y_true_train = torch.cat((y_true_train, labels.view(labels.shape[0]).detach().cpu()))

            _training_loss = running_train_loss / (train_iter + 1)
            _acc_train = accuracy_score(y_true_train, y_pred_train)

            mAP_train, MAR_train = self.calculate_map_mar(
                y_true=y_true_train,
                y_pred=y_pred_train
            )

            end_time = time.time()

            # Evaluation loop for validation set
            self.net.eval()
            running_val_loss = 0.0

            y_pred_val = torch.empty(0)
            y_true_val = torch.empty(0)
            for val_iter, data in enumerate(tqdm(validation_loader), 0):
                inputs, labels, *metadata = data

                if self.use_cuda:
                    inputs, labels = inputs.cuda('cuda:%i' % self.gpu_num), labels.cuda('cuda:%i' % self.gpu_num)

                outputs = self.net(inputs)

                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item()

                # Classification
                predicted = torch.max(outputs, 1).indices
                y_pred_val = torch.cat((y_pred_val, predicted.view(predicted.shape[0]).detach().cpu()))
                y_true_val = torch.cat((y_true_val, labels.view(labels.shape[0]).detach().cpu()))

            _val_loss = running_val_loss / (val_iter + 1)
            _acc_val = accuracy_score(y_true_val, y_pred_val)

            mAP_val, MAR_val = self.calculate_map_mar(
                y_true=y_true_val,
                y_pred=y_pred_val
            )

            # Make step for the scheduler
            if self.scheduler_type == 'ReduceLROnPlateau':
                self.scheduler.step(_val_loss)

            # Evaluation loop for test set
            running_test_loss = 0.0
            y_pred_test = torch.empty(0)
            y_true_test = torch.empty(0)
            for test_iter, data in enumerate(tqdm(test_loader), 0):
                inputs, labels, *metadata = data

                if self.use_cuda:
                    inputs, labels = inputs.cuda('cuda:%i' % self.gpu_num), labels.cuda('cuda:%i' % self.gpu_num)

                outputs = self.net(inputs)

                loss = self.criterion(outputs, labels)
                running_test_loss += loss.item()

                # Classification
                predicted = torch.max(outputs, 1).indices
                y_pred_test = torch.cat((y_pred_test, predicted.view(predicted.shape[0]).detach().cpu()))
                y_true_test = torch.cat((y_true_test, labels.view(labels.shape[0]).detach().cpu()))

            _test_loss = running_test_loss / (test_iter + 1)
            _acc_test = accuracy_score(y_true_test, y_pred_test)

            mAP_test, MAR_test = self.calculate_map_mar(
                y_true=y_true_test,
                y_pred=y_pred_test
            )

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            metrics_to_log = {
                "training loss": _training_loss,
                "validation loss": _val_loss,
                "test loss": _test_loss,

                "training mAP": mAP_train,
                "training MAR": MAR_train,
                "validation mAP": mAP_val,
                "validation MAR": MAR_val,
                "test mAP": mAP_test,
                "test MAR": MAR_test,

                "training accuracy": _acc_train,
                "val accuracy ": _acc_val,
                "test accuracy": _acc_test,

                "learning rate": current_lr,
                "epoch": epoch + 1
            }
            wandb.log(metrics_to_log)

            self.custom_print(("[Epoch: %i][Train Loss: %f][Val Loss: %f][Test Loss: %f][Time: %f]" % (
                epoch + 1, _training_loss, _val_loss, _test_loss, end_time - start_time)), log_file)

            print("\n" +
                  "torch.cuda.memory_allocated: %.2f MB" % (torch.cuda.memory_allocated(self.gpu_num) / 1024 ** 2))
            print("torch.cuda.max_memory_allocated: %.2f MB" % (
                        torch.cuda.max_memory_allocated(self.gpu_num) / 1024 ** 2))
            print("torch.cuda.memory_reserved: %.2f MB" % (torch.cuda.memory_reserved(self.gpu_num) / 1024 ** 2))
            print(
                "torch.cuda.max_memory_reserved: %.2f MB" % (torch.cuda.max_memory_reserved(self.gpu_num) / 1024 ** 2))

            free, total = torch.cuda.mem_get_info(self.gpu_num)
            print(f"free GPU memory = {free / (1024 ** 2):.2f} MB")
            print(f"total GPU memory = {total / (1024 ** 2):.2f} MB")

            # Save last trained checkpoint
            last_checkpoint_path = os.path.join(self.checkpoint_path,  "last_checkpoint_" + run_name + ".pt")
            torch.save(self.net.state_dict(), last_checkpoint_path)

            if current_lr == 0.00001 and self.scheduler_type == 'ReduceLROnPlateau':
                self.scheduler_type = 'CyclicLR'
                self.scheduler = CyclicLR(self.optimizer,
                                          base_lr=0.00001,
                                          max_lr=0.001,
                                          step_size_up=self.scheduler_step_size,
                                          step_size_down=self.scheduler_step_size,
                                          mode='triangular',
                                          cycle_momentum=self.cycle_momentum)

            # Early stopping
            early_stopping_train_loss(_training_loss, self.net)
            if early_stopping_train_loss.counter == 0:
                self.custom_print(("[Epoch: %i. Best train_loss checkpoint]" % (epoch + 1)), log_file)

            early_stopping_val_loss(_val_loss, self.net)
            if early_stopping_val_loss.counter == 0:
                self.custom_print(("[Epoch: %i. Best val_loss checkpoint]" % (epoch + 1)), log_file)

            early_stopping_val_acc(_acc_val, self.net)
            if early_stopping_val_acc.counter == 0:
                self.custom_print(("[Epoch: %i. Best validation ACC checkpoint]" % (epoch + 1)), log_file)
                self.build_confusion_matrix(
                    y_true=y_true_val,
                    y_pred=y_pred_val,
                    save_path=images_path,
                    metric_name='best_val_acc_ch(VAL_set)'
                )
                self.build_confusion_matrix(
                    y_true=y_true_train,
                    y_pred=y_pred_train,
                    save_path=images_path,
                    metric_name='best_val_acc_ch(TRAIN_set)'
                )
                self.build_confusion_matrix(
                    y_true=y_true_test,
                    y_pred=y_pred_test,
                    save_path=images_path,
                    metric_name='best_val_acc_ch(TEST_set)'
                )

            early_stopping_val_mAP(mAP_val, self.net)
            if early_stopping_val_mAP.counter == 0:
                self.custom_print(("[Epoch: %i. Best validation MAP checkpoint]" % (epoch + 1)), log_file)
                self.build_confusion_matrix(
                    y_true=y_true_val,
                    y_pred=y_pred_val,
                    save_path=images_path,
                    metric_name='best_val_mAP_ch(VAL_set)'
                )
                self.build_confusion_matrix(
                    y_true=y_true_train,
                    y_pred=y_pred_train,
                    save_path=images_path,
                    metric_name='best_val_mAP_ch(TRAIN_set)'
                )
                self.build_confusion_matrix(
                    y_true=y_true_test,
                    y_pred=y_pred_test,
                    save_path=images_path,
                    metric_name='best_val_mAP_ch(TEST_set)'
                )

            early_stopping_val_MAR(MAR_val, self.net)
            if early_stopping_val_MAR.counter == 0:
                self.custom_print(("[Epoch: %i. Best val MAR checkpoint]" % (epoch + 1)), log_file)
                self.build_confusion_matrix(
                    y_true=y_true_val,
                    y_pred=y_pred_val,
                    save_path=images_path,
                    metric_name='best_val_MAR_ch(VAL_set)'
                )
                self.build_confusion_matrix(
                    y_true=y_true_train,
                    y_pred=y_pred_train,
                    save_path=images_path,
                    metric_name='best_val_MAR_ch(TRAIN_set)'
                )
                self.build_confusion_matrix(
                    y_true=y_true_train,
                    y_pred=y_pred_train,
                    save_path=images_path,
                    metric_name='best_val_MAR_ch(TEST_set)'
                )


            early_stopping_test_MAR(MAR_test, self.net)
            if early_stopping_test_MAR.counter == 0:
                self.custom_print(("[Epoch: %i. Best test MAR checkpoint]" % (epoch + 1)), log_file)
                self.build_confusion_matrix(
                    y_true=y_true_test,
                    y_pred=y_pred_test,
                    save_path=images_path,
                    metric_name='best_test_MAR_ch(TEST_set)'
                )

            running_train_loss = 0.0
            y_pred_train = torch.empty(0)
            y_true_train = torch.empty(0)
            mAP_train, MAR_train = 0.0, 0.0

            running_val_loss = 0.0
            y_pred_val = torch.empty(0)
            y_true_val = torch.empty(0)
            mAP_val, MAR_val = 0.0, 0.0

            running_test_loss = 0.0
            y_pred_test = torch.empty(0)
            y_true_test = torch.empty(0)
            mAP_test, MAR_test = 0.0, 0.0

            if early_stopping_train_loss.early_stop:
                print(f"Early stopping")
                break

            elif early_stopping_val_loss.early_stop:
                print(f"Early stopping")
                break

            elif early_stopping_val_acc.early_stop:
                print(f"Early stopping")
                break

            elif early_stopping_val_mAP.early_stop:
                print(f"Early stopping")
                break

            elif early_stopping_val_MAR.early_stop:
                print(f"Early stopping")
                break

            elif early_stopping_test_MAR.early_stop:
                print(f"Early stopping")
                break

        log_file.close()
        wandb.finish()

        print(f'Finished Training')

    @staticmethod
    def get_loss_function(is_multilabel: bool):

        # Define loss function based on is_multilabel
        if is_multilabel:
            return torch.nn.BCEWithLogitsLoss()
        else:
            return torch.nn.CrossEntropyLoss()


    def calculate_map_mar(self, y_true: torch.tensor, y_pred: torch.tensor) -> Tuple[float, float]:
        # Calculate confusion matrix using PyTorch
        confusion_matrix_num_classes = self.num_classes
        confusion_matrix = torch.zeros((confusion_matrix_num_classes, confusion_matrix_num_classes))
        for t, p in zip(y_true, y_pred):
            confusion_matrix[t.long(), p.long()] += 1

        # Extract True Positives, False Positives, and False Negatives for each class
        TP = torch.diag(confusion_matrix)
        FP = torch.sum(confusion_matrix, dim=0) - TP
        FN = torch.sum(confusion_matrix, dim=1) - TP

        # Calculate precision and recall for each class
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        # Handle NaN values (due to 0/0) by setting them to 0
        precision[torch.isnan(precision)] = 0
        recall[torch.isnan(recall)] = 0

        # Calculate mAP and MAR
        mAP = torch.mean(precision)
        MAR = torch.mean(recall)

        return mAP.item(), MAR.item()

    def build_confusion_matrix(self, y_true: torch.tensor, y_pred: torch.tensor,
                               save_path: str, metric_name: str) -> None:
        # Convert PyTorch tensors to NumPy arrays
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        # Calculate confusion matrix using sklearn
        cm = confusion_matrix(y_true_np, y_pred_np)

        # Visualize the confusion matrix
        fig, ax = plt.subplots(figsize=(9, 9))
        cax = ax.matshow(cm, cmap="viridis")
        fig.colorbar(cax)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color='white')

        # Set the tick marks for x and y axes
        ax.set_xticks(np.arange(len(self.emotions)))
        ax.set_yticks(np.arange(len(self.emotions)))

        # Set the class names as labels
        ax.set_xticklabels(self.emotions, rotation=30)
        ax.set_yticklabels(self.emotions)

        # # Display the labels on the top
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('top')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        # Save the confusion matrix as an image
        fig_path = os.path.join(save_path, 'best_' + metric_name + '_checkpoint.png')
        plt.savefig(fig_path)
        plt.close()

    @staticmethod
    def custom_print(message, file):
        print(message)  # This will print to console
        print(message, file=file)  # This will print to the file
