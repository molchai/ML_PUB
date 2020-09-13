# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# Import PyDrive and associated libraries.
# This only needs to be done once in a notebook.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
# This only needs to be done once in a notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
from google.colab import drive
drive.mount('/gdrive')
# %cd /gdrive
import sys
sys.path.append('/gdrive/My Drive/ml/code')

import pandas as pd
import tensorflow as tf
import modelset
import numpy as np
import os
tf.keras.backend.set_floatx('float32')
os.getcwd()

class NNtranier():
  def __init__(self,batch,epochs,trainws,valws,roll_ws,name):
    self._batchsize=batch
    self._epochs=epochs
    self._trainws=trainws
    self._valws=valws
    self._roll_ws=roll_ws
    self._name=name

  def load_data(self):
    self._dataset=pd.read_pickle('My Drive/ml/data/data.pkl')

  def _get_train_data(self,start,end):
    dataset=self._dataset.query('trade_date>=@start and trade_date<@end').dropna().drop(['trade_date','stock_code'],axis=1)
    target = dataset.pop('ret')
    dataset =tf.data.Dataset.from_tensor_slices((dataset.values, target.values))
    train_dataset = dataset.shuffle(20000).batch(self._batchsize)
    return train_dataset

  def _get_val_data(self,start,end):
    dataset=self._dataset.query('trade_date>=@start and trade_date<@end').dropna().drop(['trade_date','stock_code'],axis=1)
    target = dataset.pop('ret')
    return dataset.values,target.values

  def _get_test_data(self,start, end):
    #dataset=pd.DataFrame(db.find({"trade_date":{"$gte":start, "$lt":end}},{'_id': False,'trade_date': False,'bin':False})).dropna()
    dataset=self._dataset.query('trade_date>=@start and trade_date<@end').dropna().drop(['trade_date'],axis=1)
    stock_code=dataset.pop('stock_code')
    target = dataset.pop('ret')
    return stock_code.values,dataset.values
  
  def R_squared(self,x,target_y):
    predict_y=model(x,training=False)
    return np.sum(np.square(target_y-predict_y))/np.sum(np.square(target_y))

  
  @tf.function
  def _train_step(self,x_batch_train, y_batch_train):
    with tf.GradientTape() as tape:
      y_batch_predicted = model(x_batch_train, training=True) 
      loss_value =loss_fn(y_batch_train, y_batch_predicted)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y_batch_train, y_batch_predicted)
  
  @tf.function
  def _validate_step(self,x_batch_val,y_batch_val):
    val_acc_metric.update_state(y_batch_val,model(x_batch_val,training=False))

  @property
  def _datelist(self):
    datelist=self._dataset.trade_date.unique()
    datelist=np.sort(datelist)
    return datelist

  def roll_train(self):
    datelist=self._datelist
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
      print("Restored from {}".format(manager.latest_checkpoint))
    else:
      print("Initializing from scratch.")
    train_start=0+ckpt.step.numpy()*self._roll_ws
    train_end=train_start+self._trainws
    val_start=train_end+5
    val_end=val_start+self._valws
    test_start=val_end+5
    test_end=test_start+self._roll_ws
    while test_end<len(datelist):
      train_dataset=self._get_train_data(datelist[train_start],datelist[train_end])
      x_batch_val, y_batch_val=self._get_val_data(datelist[val_start],datelist[val_end])
      summary_writer = tf.summary.create_file_writer(
      'My Drive/ml/{}/training_logs/'.format(self._name)+ datelist[train_start].astype('datetime64[s]').tolist().strftime("%Y%m%d")+'_'+datelist[train_end].astype('datetime64[s]').tolist().strftime("%Y%m%d")+'_'+datelist[val_start].astype('datetime64[s]').tolist().strftime("%Y%m%d")+'_'+datelist[val_end].astype('datetime64[s]').tolist().strftime("%Y%m%d"))
      with summary_writer.as_default():
        for epoch in range(self._epochs):
          for x_batch_train, y_batch_train in train_dataset:
            self._train_step(x_batch_train, y_batch_train)
          self._validate_step(x_batch_val,y_batch_val)
          tf.summary.scalar('tran_mse', train_acc_metric.result(), step=epoch)
          tf.summary.scalar('val_mse', val_acc_metric.result(), step=epoch)
          val_acc_metric.reset_states()
          train_acc_metric.reset_states()
      checkpoint_dir = 'My Drive/ml/{}/training_weights'.format(self._name)
      checkpoint_prefix = os.path.join(checkpoint_dir, datelist[train_start].astype('datetime64[s]').tolist().strftime("%Y%m%d")+'_'+datelist[train_end].astype('datetime64[s]').tolist().strftime("%Y%m%d"))
      model.save_weights(checkpoint_prefix)
      ckpt.step.assign_add(1)
      manager.save()
      print(datelist[train_start].astype('datetime64[s]').tolist().strftime("%Y%m%d"))
      train_start=self._roll_ws+train_start
      train_end=train_start+self._trainws
      val_start=train_end+5
      val_end=val_start+self._valws
      test_start=val_end+5
      test_end=test_start+self._roll_ws

  def roll_predict(self):
    model(np.ones((1,366)))
    datelist=self._datelist
    train_start=0
    train_end=train_start+self._trainws
    val_start=train_end+5
    val_end=val_start+self._valws
    test_start=val_end+5
    test_end=test_start+self._roll_ws
    while test_end<len(datelist):
      loc='My Drive/ml/{}/training_weights/'.format(self._name)+datelist[train_start].astype('datetime64[s]').tolist().strftime("%Y%m%d")+'_'+datelist[train_end].astype('datetime64[s]').tolist().strftime("%Y%m%d")
      model.load_weights(loc)
      for i in range(test_start,test_end):
        codes,features=self._get_test_data(datelist[i],datelist[i+1])
        predict_values=model(features,training=False)
        results=pd.DataFrame(data={"trade_date":[datelist[i]]*len(codes),"stock_code":codes,"y_hat":np.squeeze(predict_values,axis=1)})
        results.to_hdf("My Drive/ml/{}/{}.h5".format(self._name,self._name),key=self._name,mode="a",complevel=9,format='table',append=True,data_columns=['trade_date','stock_code'])
      train_start=self._roll_ws+train_start
      train_end=train_start+self._trainws
      val_start=train_end+5
      val_end=val_start+self._valws
      test_start=val_end+5
      test_end=test_start+self._roll_ws

train_acc_metric = tf.keras.metrics.MeanSquaredError()
val_acc_metric = tf.keras.metrics.MeanSquaredError()
loss_fn=tf.keras.losses.MeanSquaredError()
optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002)
model=modelset.MyModel18()
ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, 'My Drive/ml/model18/training_checkpoints', max_to_keep=3)
trainer=RNNtranier(512,200,200,20,10,'model18')
trainer.load_data()
trainer.roll_train()

model=modelset.MyModel18()
trainer=NNtranier(512,200,200,20,10,'model18')
trainer.load_data()
trainer.roll_predict()

