import argparse
import os.path
import shutil
import time

import sentencepiece as spm
import tensorflow as tf
from load_tfrecord import LoadTfRecord
from make_tfrecord import MakeTfRecord
from tensorflow.keras import mixed_precision
from transformers import GPT2Config, TFGPT2LMHeadModel


# @TODO Separate code
def train(args):  # noqa: C901
    ##########################################
    ##########################################
    # Parameter
    LEARNING_RATE = args.learning_rate
    EPSILON = args.epsilon
    CLIPNORM = args.clipnorm
    CTX_SIZE = args.ctx_size
    EMBD_SIZE = args.embd_size
    BLOCK_SIZE = args.block_size
    BATCH_SIZE = args.batch_size
    LAYER_SIZE = args.layer_size
    HEAD_SIZE = args.head_size
    NUM_EPOCH = args.num_epoch
    # Dir
    TRAIN_DIR = args.train_dir
    MODEL_DIR = args.model_dir
    OUTPUT_DIR = args.output_dir
    PREMODEL_DIR = args.premodel_dir
    CKPT_DIR = '/opt/ml/checkpoints'
    # File
    TRAIN_FILE_PATH = os.path.join(TRAIN_DIR, 'train.txt')
    TRAIN_TFRECORD_PATH = os.path.join(TRAIN_DIR, 'train.tfrecord')
    TRAIN_TFRECORD_BACKUP_PATH = os.path.join(OUTPUT_DIR, 'train.tfrecord')
    TEST_FILE_PATH = os.path.join(TRAIN_DIR, 'test.txt')
    TEST_TFRECORD_PATH = os.path.join(TRAIN_DIR, 'test.tfrecord')
    TEST_TFRECORD_BACKUP_PATH = os.path.join(OUTPUT_DIR, 'test.tfrecord')
    TOKENIZER_MODEL_PATH = os.path.join(TRAIN_DIR, 'spm.model')
    CKPT_PREFIX = os.path.join(CKPT_DIR, 'weights')
    CKPT_FILE = os.path.join(CKPT_DIR, 'checkpoint')
    ##########################################
    ##########################################

    # Using amp
    if args.fp16:
        print('####### Using AMP #######')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    # Setting tokenizer
    print('####### Loading tokenizer #######')
    tokenizer = spm.SentencePieceProcessor(TOKENIZER_MODEL_PATH)
    BOS_TOKEN_ID = tokenizer.piece_to_id('[BOS]')
    EOS_TOKEN_ID = tokenizer.piece_to_id('[EOS]')
    VOCAB_SIZE = tokenizer.get_piece_size()

    # Create TFRecord from train data and test data.
    mt = MakeTfRecord()
    if not os.path.isfile(TRAIN_TFRECORD_PATH):
        print('####### Create and Backup train TFRecord #######')
        mt.make(TRAIN_TFRECORD_PATH, TRAIN_FILE_PATH, tokenizer, BLOCK_SIZE)
        shutil.copy2(TRAIN_TFRECORD_PATH, TRAIN_TFRECORD_BACKUP_PATH)
    if not os.path.isfile(TEST_TFRECORD_PATH):
        print('####### Create and Backup test TFRecord #######')
        mt.make(TEST_TFRECORD_PATH, TEST_FILE_PATH, tokenizer, BLOCK_SIZE)
        shutil.copy2(TEST_TFRECORD_PATH, TEST_TFRECORD_BACKUP_PATH)

    print('####### Loading created TFRecord #######')
    lt = LoadTfRecord(BLOCK_SIZE)
    train_dataset = lt.load_tfrecord(TRAIN_TFRECORD_PATH, BATCH_SIZE, True)
    test_dataset = lt.load_tfrecord(TEST_TFRECORD_PATH, BATCH_SIZE)

    # GPT2 Model Config
    # See official documentation for details.
    # https://huggingface.co/transformers/model_doc/gpt2.html#gpt2config
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        n_ctx=CTX_SIZE,
        n_positions=CTX_SIZE,
        n_embd=EMBD_SIZE,
        n_layer=LAYER_SIZE,
        n_head=HEAD_SIZE,
        # When setting `use_cache=True`,
        # an error will occur unless `BATCH_SIZE` and `HEAD_SIZE` are equalized
        use_cache=False,
    )

    # Multiple GPU setting.
    # See the official Tensorflow documentation for more details.
    # https://www.tensorflow.org/guide/distributed_training#mirroredstrategy
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        if PREMODEL_DIR is not None:
            print('####### Load model from pretrained #######')
            model = TFGPT2LMHeadModel.from_pretrained(
                PREMODEL_DIR, use_cache=False
            )
        else:
            print('####### Create new model #######')
            model = TFGPT2LMHeadModel(config)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, epsilon=EPSILON, clipnorm=CLIPNORM
        )

        # @NOTE: Default initial_scale is 2**15
        # The default value is too large and becomes inf, so I set it small.
        # Consider reverting to the default if necessary.
        # See the official Tensorflow documentation for more details.
        # https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer
        optimizer = mixed_precision.LossScaleOptimizer(
            optimizer, initial_scale=2**12
        )

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy'
        )
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy'
        )

        # @FIXME
        # custom loss function
        # When using AMP, somehow SparseCategoricalCrossentropy causes NaN.
        # I cast logits and run Sparse Categorical Crossentropy at float32
        # to prevent NaN from occurring.
        # If anyone knows the fundamental solution, please let me know.
        def loss_fn(labels, logits):
            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )
            if args.fp16:
                loss = loss_obj(
                    labels, tf.cast(logits, tf.float32)
                )
            else:
                loss = loss_obj(labels, logits)
            return loss

        # Based on transformers' TFCausalLanguageModelingLoss.compute_loss.
        # https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_tf_utils.py#L151
        def compute_loss(labels, logits):
            dynamic = tf.shape(logits)
            if logits.shape == tf.TensorShape(None):
                return dynamic
            static = logits.shape.as_list()
            shape_list = [
                dynamic[i] if s is None else s for i, s in enumerate(static)
            ]

            active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
            reduced_logits = tf.boolean_mask(
                tf.reshape(logits, (-1, shape_list[2])), active_loss
            )
            new_labels = tf.boolean_mask(
                tf.reshape(labels, (-1,)), active_loss
            )

            per_example_loss = loss_fn(new_labels, reduced_logits)
            per_example_loss /= tf.cast(
                tf.reduce_prod(tf.shape(labels)[1:]), tf.float32
            )
            return tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=BATCH_SIZE
            )

    # Custom Training Loop Definition
    def train_step(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            predictions = model(x).logits
            loss = compute_loss(y, predictions)
            scaled_loss = optimizer.get_scaled_loss(loss)
        scaled_gradients = tape.gradient(
            scaled_loss, model.trainable_variables
        )
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(y, predictions)
        return loss

    def test_step(inputs):
        x, y = inputs
        predictions = model(x, training=False).logits
        t_loss = loss_fn(y, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(y, predictions)

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = mirrored_strategy.run(
            train_step, args=(dataset_inputs,)
        )
        return mirrored_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )

    @tf.function
    def distributed_test_step(dataset_inputs):
        return mirrored_strategy.run(test_step, args=(dataset_inputs,))

    if os.path.isfile(CKPT_FILE):
        print('####### Restore ckpt #######')
        checkpoint.restore(tf.train.latest_checkpoint(CKPT_DIR))

    print("####### Start training #######")
    dist_train_dataset = mirrored_strategy.experimental_distribute_dataset(
        train_dataset
    )
    dist_test_dataset = mirrored_strategy.experimental_distribute_dataset(
        test_dataset
    )
    for epoch in range(NUM_EPOCH):
        print("\nStart of epoch %d" % (epoch + 1,))
        start_time = time.time()

        total_loss = 0.0
        total_steps = 0
        for step, x in enumerate(dist_train_dataset, 1):
            total_loss += distributed_train_step(x)
            total_steps = step
            if step % 5000 == 0:
                print(
                    "Training loss at step %d: %.4f"
                    % (step, (total_loss / step))
                )
        train_loss = total_loss / total_steps
        for x in dist_test_dataset:
            distributed_test_step(x)

        # Save ckpt at every epoch
        checkpoint.save(CKPT_PREFIX)

        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                    "Test Accuracy: {}")
        print(template.format(epoch + 1,
                              train_loss,
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        print("Time taken: %.2fs" % (time.time() - start_time))

    model.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--ctx_size", type=int, default=1024)
    parser.add_argument("--embd_size", type=int, default=768)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--layer_size", type=int, default=12)
    parser.add_argument("--head_size", type=int, default=12)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--fp16", action='store_true')

    parser.add_argument(
        "--model_dir", type=str, default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument(
        '--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR']
    )
    parser.add_argument(
        '--premodel_dir', type=str, default=os.environ.get('SM_CHANNEL_PREMODEL')
    )

    args, _ = parser.parse_known_args()

    train(args)
