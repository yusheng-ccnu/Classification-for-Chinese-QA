import pylab as pl
from IPython import display
from keras.callbacks import Callback


class DrawCallback(Callback):
    def __init__(self, runtime_plot=True):
        super().__init__()
        self.init_loss = None
        self.runtime_plot = runtime_plot

        self.xdata = []
        self.ydata = []

    def _plot(self, epoch=None):
        epochs = self.params.get("epochs")
        pl.ylim(0, int(self.init_loss * 2))
        pl.xlim(0, epochs)

        pl.plot(self.xdata, self.ydata)
        pl.xlabel('Epoch {}/{}'.format(epoch or epochs, epochs))
        pl.ylabel('Loss {:.4f}'.format(self.ydata[-1]))

    def _runtime_plot(self, epoch):
        self._plot(epoch)

        display.clear_output(wait=True)
        display.display(pl.gcf())
        pl.gcf().clear()

    def plot(self):
        self._plot()
        pl.show()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if self.init_loss is None:
            self.init_loss = loss
        self.xdata.append(epoch)
        self.ydata.append(loss)
        if self.runtime_plot:
            self._runtime_plot(epoch)
