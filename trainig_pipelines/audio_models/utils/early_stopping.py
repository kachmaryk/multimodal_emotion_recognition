import torch


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience: int = 30, verbose: bool = False,
                 delta: float = 0.0001, path: str = 'checkpoint.pt',
                 trace_func=print, is_loss: bool = False,
                 name_of_metric: str = 'Val loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 30
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float):  Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.0001
            path (str):     Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func:     trace print function.
            (function)      Default: print
        """

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.is_loss = is_loss
        self.name_of_metric = name_of_metric

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric, model):
        score = metric
        if self.is_loss:
            score = -score

        if self.best_score is None:
            self.save_checkpoint(model, self.best_score, score)
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.patience:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

        else:
            self.save_checkpoint(model, self.best_score, score)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, model, previous_metric, new_metric):
        """
        Saves model when metric improves.
        """
        if self.verbose:
            if self.is_loss:
                previous_metric, new_metric = -previous_metric, -new_metric
                self.trace_func(f'{self.name_of_metric} decreased: ({previous_metric:.4f} --> {new_metric:.4f}).',
                                '   Saving model ...')
            else:
                self.trace_func(f'{self.name_of_metric} increased: ({previous_metric:.4f} --> {new_metric:.4f}).',
                                '   Saving model ...')

        torch.save(model.state_dict(), self.path)
