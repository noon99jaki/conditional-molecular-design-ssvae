from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import pandas as pds

from preprocessing import smiles_to_seq, vectorize
import SSVAE
from preprocessing import get_property, canonocalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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
char_set=[' ','1','2','3','4','5','6','7','8','9','-','#','(',')','[',']','+','=','B','Br','c','C','Cl','F','H','I','N','n','O','o','P','p','S','s','Si','Sn']
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
else:
    ntrn=300000
    ntst=10000
    frac_val=0.05
    dim_z = 100
    dim_h = 250
    n_hidden = 3
    batch_size = 200

# data preparation
print('::: data preparation')

smiles = pds.read_csv(data_uri).as_matrix()[:ntrn+ntst,0] #0: SMILES
Y = np.asarray(pds.read_csv(data_uri).as_matrix()[:ntrn+ntst,1:], dtype=np.float32) # 1: MolWT, 2: LogP, 3: QED 

list_seq = smiles_to_seq(smiles, char_set)
Xs, X=vectorize(list_seq, char_set)
seqlen_x = X.shape[1]
dim_x = X.shape[2]
dim_y = Y.shape[1]

model = SSVAE.Model(seqlen_x = seqlen_x, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, dim_h = dim_h,
                    n_hidden = n_hidden, batch_size = batch_size, beta = float(beta), char_set = char_set,
                    save_uri = save_uri)

scaler_Y = StandardScaler()
scaler_Y.fit(Y)

with model.session:
    ## model training
    print('::: model training')

    model.trainXY(X, Xs, Y, scaler_Y, ntrn, ntst, frac, frac_val, experiment_dir)

    ## property prediction performance
    tstY_hat=scaler_Y.inverse_transform(model.predict(tstX))

    for j in range(dim_y):
        print([j, mean_absolute_error(tstY[:,j], tstY_hat[:,j])])
        
        
    ## unconditional generation
    for t in range(10):
        smi = model.sampling_unconditional()
        print([t, smi, get_property(smi)])
    
    ## conditional generation (e.g. MolWt=250)
    yid = 0
    ytarget = 250.
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])
    
    for t in range(10):
        smi = model.sampling_conditional(yid, ytarget_transform)
        print([t, smi, get_property(smi)])