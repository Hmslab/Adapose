import sys
sys.path.append('./PointNet/')
sys.path.append('./tf_ops/sampling/')
sys.path.append('./tf_ops/grouping')
sys.path.append('./tf_ops/3d_interpolation/')

import os
import argparse
import model_proj_weights as model
import tensorflow as tf

import scipy.io as io
import numpy as np

def main():
    # indx = np.arange(200)

    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', help='Path to dataset', required=True)
    parser.add_argument('--save_path', '-s', help='Path to folder for saving check points and summary', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    parser.add_argument('--is_training', '-t', help='Training or Not', default=True)
    parser.add_argument('--g_lr', '-g_lr', help='learning rate for generator', default=1e-4)
    parser.add_argument('--d_lr', '-d_lr', help='learning rate for discriminator', default=1e-4)
    parser.add_argument('--max_iteration', '-i', help='Number of max training iter (default defined in setting)', default=1e4, type=int)
    parser.add_argument('--mode', '-m', help='train, test or demo', required=True)
    args = parser.parse_args()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.InteractiveSession(config=config)

    m = model.Trainer(args, sess, 'test')
    if args.mode == 'train':
        m.train()
    elif args.mode == 'test':
        m.test()
    elif args.mode == 'demo':
        m.demo()
    elif args.mode == 'performance':
        m.performance_test()
    


if __name__ == '__main__':
    main()
