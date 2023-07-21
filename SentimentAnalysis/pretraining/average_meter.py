class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count_loss = None
        self.avg_loss = None
        self.sum_loss = None
        self.val_loss = None
        self.count_accuracy = None
        self.avg_accuracy = None
        self.sum_accuracy = None
        self.val_accuracy = None
        self.reset()

    def reset(self):
        self.val_loss = 0
        self.avg_loss = 0
        self.sum_loss = 0
        self.count_loss = 0
        self.count_accuracy = 0
        self.avg_accuracy = 0
        self.sum_loss = 0
        self.val_loss = 0

    def update_loss(self, val, n=1):
        self.val_loss = val
        self.sum_loss += val * n
        self.count_loss += n
        self.avg_loss = self.sum_loss / self.count_loss

    def update_accuracy(self, val, n=1):
        self.val_accuracy = val * 100
        self.sum_accuracy += val * n
        self.count_accuracy += n
        self.avg_accuracy = self.sum_accuracy * 100 / self.count_accuracy
