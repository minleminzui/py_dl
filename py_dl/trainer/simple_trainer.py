from .trainer import Trainer


class SimpleTrainer(Trainer):

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)

    def _variable_weights_init(self):
        """not in distribbuted training, so use the initilization method of the node itself"""
        pass

    def _optimizer_update(self):
        self.optimizer.update()
