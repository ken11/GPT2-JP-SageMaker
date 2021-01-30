import tensorflow as tf


class LoadTfRecord:

    def __init__(self, block_size):
        self.block_size = block_size

    def load_tfrecord(self, path, batch_size, shuffle=False):
        dataset = tf.data.TFRecordDataset(
            [path]
        ).map(
            self._parse_function
        ).batch(
            batch_size, drop_remainder=True
        )

        if shuffle:
            dataset = dataset.shuffle(self._count(path))

        return dataset

    def _count(self, path):
        # @FIXME Deprecated
        return len(list(tf.compat.v1.io.tf_record_iterator(path)))

    def _parse_function(self, example_proto):
        feature_description = {
            'inputs': tf.io.FixedLenFeature((self.block_size - 1), tf.int64),
            'labels': tf.io.FixedLenFeature((self.block_size - 1), tf.int64),
        }
        x = tf.io.parse_single_example(example_proto, feature_description)
        return (x['inputs'], x['labels'])
