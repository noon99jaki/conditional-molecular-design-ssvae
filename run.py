from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import pandas as pds

from preprocessing import ZINC
import SSVAE

# FLAGS
FLAGS = tf.app.flags.FLAGS  


# Experiment name
tf.flags.DEFINE_string('output_dir', './output', 'output folder.')
tf.flags.DEFINE_string('experiment_name', 'dbg300',
                       'All outputs of this experiment is'
                       ' saved under a folder with the same name.')
tf.app.flags.DEFINE_bool('debug', True, 'debug mode.')

experiment_dir = os.path.join(FLAGS.output_dir, FLAGS.experiment_name)
if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MkDir(FLAGS.output_dir)
if not tf.gfile.IsDirectory(experiment_dir):
    tf.gfile.MkDir(experiment_dir)

# pre-defined parameters
frac=0.5
beta=10000.
data_uri='./data/ZINC_310k.csv'
save_uri=os.path.join(experiment_dir, 'models', 'model.ckpt')

debug=True

if debug:
    ntrn=300
    ntst=100
    frac_val=0.1
    dim_z = 10
    dim_h = 25
    n_hidden = 2
    batch_size = 10
    max_epoch = 10
else:
    ntrn=300000
    ntst=10000
    frac_val=0.05
    dim_z = 100
    dim_h = 250
    n_hidden = 3
    batch_size = 200
    max_epoch = 300

# data preparation
print('::: data preparation')
smiles = pds.read_csv(data_uri).as_matrix()[:ntrn+ntst,0] #0: SMILES
Y = np.asarray(pds.read_csv(data_uri).as_matrix()[:ntrn+ntst,1:], dtype=np.float32) # 1: MolWT, 2: LogP, 3: QED 
list_seq = ZINC.smiles_to_seq(smiles)
Xs, X=ZINC.vectorize(list_seq)
seqlen_x = X.shape[1]
dim_x = X.shape[2]
dim_y = Y.shape[1]

model = SSVAE.Model(seqlen_x = seqlen_x, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, dim_h = dim_h,
                    n_hidden = n_hidden, batch_size = batch_size, beta = float(beta), char_set = ZINC.char_set,
                    save_uri = save_uri)

with model.session:
    ## model training
    print('::: model training')

    model.trainXY(max_epoch, X, Xs, Y, ntrn, ntst, frac, frac_val, experiment_dir)

    ## unconditional generation
    for t in range(10):
        smi = model.sampling_unconditional()
        print([t, smi, ZINC.get_property(smi)])
    
    ## conditional generation (e.g. MolWt=250)
    yid = 0
    ytarget = 250.

    for t in range(10):
        smi = model.sampling_conditional_transform(yid, ytarget)
        print([t, smi, ZINC.get_property(smi)])