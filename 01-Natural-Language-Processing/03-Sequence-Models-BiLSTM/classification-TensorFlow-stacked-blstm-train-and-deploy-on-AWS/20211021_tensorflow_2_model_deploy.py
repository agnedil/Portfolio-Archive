#!/usr/bin/env python
# coding: utf-8

# SCRIPT TO DEPLOY TENSORFLOW 2 MODEL

import sagemaker
import boto3
from sagemaker.tensorflow.model import TensorFlowModel
import numpy as np
import json
#s3 = boto3.resource('s3')


'''
# SAVE MODEL IN CORRECT FILE FORMAT

import tensorflow as tf
import os

# LOAD TRAINING EPOCH
model_path = 'model-deployment/bilstm-embeds/20211005T2010-e10.hdf5'
model = tf.keras.models.load_model( model_path )

# SAVE IT IN SAVE_MODEL FORMAT IN SUBDIRECTORY '1'
filepath = os.path.join('1')
model.save(filepath)

# ARCHIVE SAVED MODEL FOR DEPLOYMENT
os.system('tar -czvf model.tar.gz 1/')
'''

# DEPLOY MODEL

# 'model.tar.gz' and tokenizer ('20211005T1859_tokenizer.pkl' as loaded in inference.py) are in the same S3 folder as 'model_data' below
# 'inference.py' and 'bb_classifier_inference.py' are in the same location from which this deploy script is being run
# normally, 'source_dir' is used to indicated the location of the last 2 files on S3, but it's not working in this case for some reason
tf_model = TensorFlowModel(
    role        = '',
    model_data  = 's3:// /model.tar.gz',
    #source_dir  = 's3://',
    entry_point = './inference.py',
    framework_version = '2.4.1',
    dependencies = [ './bb_classifier_inference.py' ],
)

# CREATE ENDPOINT
endpoint = 'test-bin-bias-model-00'
tf_model.deploy(
    initial_instance_count = 1,
    instance_type = 'ml.g4dn.xlarge',
    endpoint_name = endpoint,
    wait = True,
    update_endpoint = True,
    #aws_region='eu-central-1',
)

# TEST DEPLOYED MODEL AT NEWLY CREATED ENDPOINT
endpoint          = endpoint
sagemaker_session = sagemaker.Session()
predictor         = sagemaker.tensorflow.model.TensorFlowPredictor(
                                                                    endpoint,
                                                                    sagemaker_session
                                                                  )

# THIS SHOULD OUTPUT: {'predictions': [[0.99979347], [0.999683857], [0.53700906], [0.00351781468]]}
input_sentences = [
    'Considering you are the only person at the counter today who speaks Spanish you have been very helpful to all other crewmembers with interpreting',
    'You always jump in to help with no questions asked',
    "Can't find any bias here - add me to list of positive examples",
    'Thank you for being a wonderful partner!',
]

print(predictor.predict(input_sentences))
