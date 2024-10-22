import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

mlflow.set_experiment("odpy")

#add config path
pipeline_config_path = 'path/to/pipeline.config'    
 


config = config_util.get_configs_from_pipeline_file(pipeline_config_path)
model_config = config['model']
train_config = config['train_config']
train_input_config = config['train_input_config']
eval_input_config = config['eval_input_config']

def train_and_log_model():
    with mlflow.start_run():
        
        learning_rate = 0.001
        batch_size = 4
        num_steps = 5000

        
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_steps", num_steps)

        
        model = tf.keras.models.load_model(checkpoint_dir)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
    
        train_dataset = tf.data.TFRecordDataset(filenames=[''])
        val_dataset = tf.data.TFRecordDataset(filenames=['odpy/odpy/odpyFabric.v8i.tfrecord.zip'])

    
        model.fit(train_dataset, validation_data=val_dataset, epochs=10, batch_size=batch_size)

        
        mlflow.tensorflow.log_model(tf_saved_model_dir=checkpoint_dir, artifact_path="model")

        
        model.save("odpy/odpy")

if __name__ == "__main__":
    train_and_log_model()


