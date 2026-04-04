import os
import time
import tensorflow as tf

from CNNClassifier.entity import PrepareCallbacksConfig
from livelossplot.inputs.keras import PlotLossesCallback


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True,
            verbose=1
        )
        return checkpoint

    @property
    def _create_es_callbacks(self):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            mode='min'
        )
        return early_stop

    @property
    def _create_plot_loss(self):
        return PlotLossesCallback()

    def callbacks(self, plot=False):
        if plot:
            cb = [
                self._create_tb_callbacks,
                self._create_ckpt_callbacks,
                self._create_es_callbacks,
                self._create_plot_loss,
            ]
        else:
            cb = [
                self._create_tb_callbacks,
                self._create_ckpt_callbacks,
                self._create_es_callbacks,
            ]
        return cb
