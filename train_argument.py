from datetime import datetime
import tensorflow as tf


input_size = [64,64,3]

learning_rate_init = 10 ** -3
num_epochs = 100
batch_size = 64

folder = 'result/'+ datetime.today().strftime(f'%Y_%m_%d_%H_%M_%S_lr_{learning_rate_init}_ep_{num_epochs}_batch_{batch_size}')

# model_four_tasks
model_input = 5
model_consecutive = 5
model_resnet = 5

# pipeline
#data_csv_path = 'data/image_path.csv'
data_csv_path = 'data/test_path.csv'
json_path = 'data/json/'
object_count = 10 * 4 # x,y,w,h 때문에 4를 곱해준다
num_samples = 3


#tf_config = tf.config.experimental()