# License
# Copyright 2018 Hamaad Musharaf Shah
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

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