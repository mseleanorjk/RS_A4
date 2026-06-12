# Callbacks

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, warmup_epochs=0):
        self.patience = patience
        self.delta = delta
        self.warmup_epochs = warmup_epochs
        self.previous_recall10 = float('-inf')
        self.counter = 0
        self.best_weights = None

    def step(self, recall10, epoch):
        if epoch < self.warmup_epochs:
            return False  # don't stop during warmup
        if recall10 is None:
            return False 

        if recall10 > self.previous_recall10 - self.delta and recall10 < self.previous_recall10 + self.delta:
            self.counter += 1
        else:
            self.previous_recall10 = recall10

        return self.counter >= self.patience  # True = stop training


class EarlyStoppingLoss:
    def __init__(self, patience=5, delta=0.0, warmup_epochs=0):
        self.patience = patience
        self.delta = delta
        self.warmup_epochs = warmup_epochs
        self.previous_val_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def step(self, val_loss, epoch):
        if epoch < self.warmup_epochs:
            return False  # don't stop during warmup

        if val_loss + self.delta > self.previous_val_loss:
            self.counter += 1
        else:
            self.previous_val_loss = val_loss

        return self.counter >= self.patience  # True = stop training