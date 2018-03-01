import os
import tensorflow as tf

def get_session(gpu_fraction=0.25,
                allow_soft_placement=True,
                log_device_placement=True):
    """
    https://groups.google.com/forum/#!topic/keras-users/MFUEY9P1sc8
    """
    num_threads = os.environ.get("OMP_NUM_THREADS")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads, allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement))