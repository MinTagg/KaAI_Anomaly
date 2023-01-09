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
data_csv_path = 'data/for_test/json_test.csv'

#json_path = 'data/json/'
json_path = 'data/for_test/json/'
resnet_path = 'data/resnet/'
object_count = 10 * 4 # x,y,w,h 때문에 4를 곱해준다
num_samples = 3
resnet_output = (7,7,2048)

# Optical Flow
flow_size = (1280, 720, 2)
flow_path = 'data/for_test/flow/frame'
flow_size = (256,256)

# image
image_path = 'data/for_test/image/frame'

# anomaly probability
prob = 0.3 # 30% 확률로 이상 데이터를 생성한다
# anomaly task list
# task 3번 middle box prediction은 변형이 불가능한 task임
task_list = [1,2]


#tf_config = tf.config.experimental()