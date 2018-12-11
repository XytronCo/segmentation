import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

slim = tf.contrib.slim

#============INPUT ARGUMENTS================
flags = tf.app.flags

#Directories
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('checkpoint_dir', './log/train_original_ENet', 'The checkpoint directory to restore your mode.l')
flags.DEFINE_string('logdir', './log/original_test', 'The log directory for event files created during test evaluation.')

#Evaluation information
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
#flags.DEFINE_integer('batch_size', 10, 'The batch_size for evaluation.')
flags.DEFINE_integer('batch_size', 1, 'The batch_size for evaluation.')
flags.DEFINE_integer('image_height', 360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
#flags.DEFINE_integer('num_epochs', 10, "The number of epochs to evaluate your model.")
flags.DEFINE_integer('num_epochs', 1, "The number of epochs to evaluate your model.")

#Architectural changes
flags.DEFINE_integer('num_initial_blocks', 1, 'The number of initial blocks to use in ENet.')
flags.DEFINE_integer('stage_two_repeat', 2, 'The number of times to repeat stage two.')
flags.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')

FLAGS = flags.FLAGS

#==========NAME HANDLING FOR CONVENIENCE==============
num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size
image_height = FLAGS.image_height
image_width = FLAGS.image_width
num_epochs = FLAGS.num_epochs

#Architectural changes
num_initial_blocks = FLAGS.num_initial_blocks
stage_two_repeat = FLAGS.stage_two_repeat
skip_connections = FLAGS.skip_connections

dataset_dir = FLAGS.dataset_dir
checkpoint_dir = FLAGS.checkpoint_dir
logdir = FLAGS.logdir

#===============PREPARATION FOR TRAINING==================
#Checkpoint directories
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

#=============EVALUATION=================
def run():
    with tf.Graph().as_default() as graph:
	
	image_name = sys.argv[1]
	images = tf.convert_to_tensor([image_name])
	input_queue = tf.train.slice_input_producer([images])

        #Decode the image and annotation raw content
        image = tf.read_file(input_queue[0])
        image = tf.image.decode_image(image, channels=3)


        #Create the model inference
	image = preprocess(image, None, image_height, image_width)
		
        image = tf.train.batch([image], batch_size=batch_size, allow_smaller_final_batch=True)


        #Create the model inference
        with slim.arg_scope(ENet_arg_scope()):
            logits, probabilities = ENet(image,
                                         num_classes,
                                         batch_size=batch_size,
                                         is_training=True,
                                         reuse=None,
                                         num_initial_blocks=num_initial_blocks,
                                         stage_two_repeat=stage_two_repeat,
                                         skip_connections=skip_connections)

        # Set up the variables to restore and restoring function from a saver.
        exclude = []
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)


        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(probabilities, -1)


        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = logdir, summary_op = None, init_fn=restore_fn)

        #Run the managed session
        with sv.managed_session() as sess:
                logging.info('Saving the images now...')
                predictions_val = sess.run(predictions)
		img = predictions_val[0]
		plt.imshow(img)
		plt.axis('off'), plt.xticks([]), plt.yticks([])
		plt.tight_layout()
		plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
		out_file_name = "./output/result"+sys.argv[1].split('.')[0]
		plt.savefig(out_file_name, bbox_inces='tight', pad_inches=0)
		print(out_file_name)

if __name__ == '__main__':
    run()
