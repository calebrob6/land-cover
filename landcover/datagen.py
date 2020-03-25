import numpy as np
import keras.utils

from utils import handle_labels, classes_in_key, to_float
from helpers import get_logger

logger = get_logger(__name__)


def color_aug(colors):
    n_ch = colors.shape[0]
    contra_adj = 0.05
    bright_adj = 0.05

    ch_mean = np.mean(colors, axis=(-1, -2), keepdims=True).astype(np.float32)

    contra_mul = np.random.uniform(1 - contra_adj, 1 + contra_adj, (n_ch, 1, 1)).astype(
        np.float32
    )
    bright_mul = np.random.uniform(1 - bright_adj, 1 + bright_adj, (n_ch, 1, 1)).astype(
        np.float32
    )

    colors = (colors - ch_mean) * contra_mul + ch_mean * bright_mul
    return colors


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    # pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
    def __init__(
        self,
        patches,
        batch_size,
        steps_per_epoch,
        input_size,
        output_size,
        num_channels=4,
        num_classes=5,
        lr_num_classes=22,
        hr_labels_index=8,
        lr_labels_index=9,
        hr_label_key="data/cheaseapeake_to_hr_labels.txt",
        lr_label_key="data/nlcd_to_lr_labels.txt",
        do_color_aug=False,
        do_superres=False,
        superres_only_states=(),
        data_type="uint16",
    ):
        """Initialization"""

        self.patches = patches
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        assert steps_per_epoch * batch_size < len(patches)

        self.input_size = input_size
        self.output_size = output_size

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.lr_num_classes = lr_num_classes

        self.do_color_aug = do_color_aug

        self.do_superres = do_superres
        self.superres_only_states = superres_only_states
        self.on_epoch_end()

        self.hr_labels_index = (hr_labels_index,)
        self.lr_labels_index = lr_labels_index

        self.hr_label_key = hr_label_key
        self.lr_label_key = lr_label_key

        self.data_type = data_type

        if self.hr_label_key:
            assert self.num_classes == classes_in_key(self.hr_label_key)
        if self.lr_label_key:
            assert self.lr_num_classes == classes_in_key(self.lr_label_key)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.steps_per_epoch

    def __getitem__(self, index):
        """Generate one batch of data"""
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        fns = [self.patches[i] for i in indices]

        x_batch = np.zeros(
            (self.batch_size, self.input_size, self.input_size, self.num_channels),
            dtype=np.float32,
        )
        y_hr_batch = np.zeros(
            (self.batch_size, self.output_size, self.output_size, self.num_classes),
            dtype=np.float32,
        )
        y_sr_batch = None
        if self.do_superres:
            y_sr_batch = np.zeros(
                (
                    self.batch_size,
                    self.output_size,
                    self.output_size,
                    self.lr_num_classes,
                ),
                dtype=np.float32,
            )

        for i, (fn, state) in enumerate(fns):
            if fn.endswith(".npz"):
                dl = np.load(fn)
                data = dl["arr_0"].squeeze()
                dl.close()
            elif fn.endswith(".npy"):
                data = np.load(fn).squeeze()
            data = np.rollaxis(data, 0, 3)

            # do a random crop if input_size is less than the prescribed size
            assert data.shape[0] == data.shape[1]
            data_size = data.shape[0]
            if self.input_size < data_size:
                x_idx = np.random.randint(0, data_size - self.input_size)
                y_idx = np.random.randint(0, data_size - self.input_size)
                data = data[
                    y_idx : y_idx + self.input_size, x_idx : x_idx + self.input_size, :
                ]

            x_batch[i] = to_float(data[:, :, : self.num_channels], self.data_type)

            # setup x
            if self.do_color_aug:
                x_batch[i] = color_aug(x_batch[i])

            # setup y_highres
            if self.hr_label_key:
                y_train_hr = handle_labels(
                    data[:, :, self.hr_labels_index], self.hr_label_key
                )
            else:
                y_train_hr = data[:, :, self.hr_labels_index]
            y_train_hr = keras.utils.to_categorical(y_train_hr, self.num_classes)

            if self.do_superres:
                if state in self.superres_only_states:
                    y_train_hr[:, :, 0] = 0
                else:
                    y_train_hr[:, :, 0] = 1
            else:
                y_train_hr[:, :, 0] = 0
            y_hr_batch[i] = y_train_hr

            # setup y_superres
            if self.do_superres:
                if self.lr_label_key:
                    y_train_nlcd = handle_labels(
                        data[:, :, self.lr_labels_index], self.lr_label_key
                    )
                else:
                    y_train_nlcd = data[:, :, self.lr_labels_index]
                y_train_nlcd = keras.utils.to_categorical(
                    y_train_nlcd, self.lr_num_classes
                )
                y_sr_batch[i] = y_train_nlcd

        if self.do_superres:
            return x_batch.copy(), {"outputs_hr": y_hr_batch, "outputs_sr": y_sr_batch}
        else:
            return x_batch.copy(), y_hr_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.patches))
        np.random.shuffle(self.indices)
