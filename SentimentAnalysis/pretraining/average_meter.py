class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count_loss = 0
        self.avg_loss = 0
        self.sum_loss = 0
        self.val_loss = 0
        self.count_accuracy = 0
        self.avg_accuracy = 0
        self.sum_accuracy = 0
        self.val_accuracy = 0
        self.reset()

    def reset(self):
        self.count_loss = 0
        self.avg_loss = 0
        self.sum_loss = 0
        self.val_loss = 0
        self.count_accuracy = 0
        self.avg_accuracy = 0
        self.sum_accuracy = 0
        self.val_accuracy = 0

    def update_loss(self, val, n=1):
        self.val_loss = val
        self.sum_loss += val * n
        self.count_loss += n
        self.avg_loss = self.sum_loss / self.count_loss

    def update_accuracy(self, val, n=1):
        self.val_accuracy = (val * 100) / n
        self.sum_accuracy += val
        self.count_accuracy += n
        self.avg_accuracy = self.sum_accuracy * 100 / self.count_accuracy
