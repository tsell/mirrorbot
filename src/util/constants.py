import tensorflow as tf

FRAME_DTYPE = tf.uint8
FRAME_SHAPE = [640,480,3]

ACTION_DICT = {
    1: 'left',
    2: 'right'
}
ACTION_VOCAB_SIZE = len(ACTION_DICT)

