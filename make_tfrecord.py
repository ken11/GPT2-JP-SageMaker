import tensorflow as tf


class MakeTfRecord:

    def make(self, tfrecord_path, path, tokenizer, block_size):
        texts = self._load_file(path)
        tokenized = self._tokenize(tokenizer, texts)
        self._write_tfrecord(tfrecord_path, tokenized, block_size)

    def _load_file(self, path):
        texts = []
        for line in open(path, "r", encoding='utf-8'):
            texts.append(line)
        return texts

    def _tokenize(self, tokenizer, texts):
        tokenized = []
        for text in texts:
            tokenized.extend(tokenizer.encode(text))
        return tokenized

    def _write_tfrecord(self, path, tokenized, block_size):
        examples = []
        for i in range(0, len(tokenized) - block_size + 1, block_size):
            examples.append(tokenized[i:i + block_size])

        with tf.io.TFRecordWriter(path) as writer:
            for ex in examples:
                inputs = (ex[:-1])
                labels = (ex[1:])
                writer.write(self._serialize(inputs, labels))

    def _serialize(self, inputs, labels):
        feature = {
            'inputs': self._int64_feature(inputs),
            'labels': self._int64_feature(labels),
        }
        proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return proto.SerializeToString()

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
