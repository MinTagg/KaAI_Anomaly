import json
import numpy as np
import tensorflow as tf
import pandas as pd
import train_argument as arg
import os
import cv2
from skimage.transform import resize
import random

class ano_dataset(tf.data.Dataset):
    def _generator():
        margin = (arg.model_input-1)//2
        df = pd.read_csv(arg.data_csv_path)
        df = df.sample(frac=1) # 데이터 프레임 셔플
        for _, row in df.iterrows(): # csv 파일에서 하나의 줄을 읽어온다
            _, name = row['Path'].split('/') # name = 1550.jpg
            json_file_name = []
            json_files = [] # 해당 범위의 모든 json 파일을 불러와서 해당 리스트에 저장한다
            mid_index = arg.model_input - margin - 1
            state = True
            ########## 모든 파일이 json 파일이 있는지 확인해야 한다
            ########## 없다면 다음 줄로 확인할 수 있도록
            for i in range(arg.model_input): # 열기 위한 파일의 이름들을 저장한다
                json_file_name.append(f'{arg.json_path+str(int(name[:-4])-margin+i)}.json')
                if os.path.isfile(json_file_name[-1]) == False: # 해당 파일이 존재하지 않는다면 state를 False, 다음 csv 파일을 읽는다
                    state = False
                    break
            if state: # 범위의 모든 json 파일이 존재하는 경우
                # 모든 json 파일을 열어보면서 공통의 key를 찾는다
                for i in json_file_name:
                    with open(i, 'r') as f: # json_files에 각 파일들을 읽어서 불러옴
                        json_files.append(json.load(f)) # json파일을 저장한다
                main_keys = json_files[mid_index].keys() # 중간 파일의 key들을 main_key에 저장한다 -> 확인
                
                for key in main_keys: # 각 key들을 하나씩 검사 -> 모든 파일에 key가 존재하는지 확인한다
                    

                #### key filter 추가 구역 :: if key in target_list:


                    state = True
                    result = [] # bbox 정보를 저장할 list -> 각 key별로 저장
                    for instance in json_files: # 모든 파일
                        if key not in instance.keys(): # -> 없는 key 검출 확인 완료
                            state = False # key가 없다면 더이상 저장하지 않고 넘긴다
                            break
                    if state: # 모든 파일에 key가 존재하는 경우
                        for file in json_files:
                            result += file[key]['bbox'] # bbox를 result에 저장한다

                        print(f'Mid File :: {json_file_name[mid_index]}')
                        print(f'Key Value :: {key}')
                #for i in range(len(result)):
                #    print(result[i], end=' ')
                #    if (i+1)%4==0:
                #        print('')
                #print('')
                # 0. if state일 경우에 optical flow 파일을 불러온다
                ##### 이름을 기반으로 하는 for문 시작
                    # 1. 이름따라 optical flow 파일을 불러온다
                    # 2. 한개의 optical flow 파일을 불러오고, 해당 파일에서 bbox에 해당하는 부분을 잘라낸다
                    # 3. 잘라낸 optical flow 파일을 skitlearn의 resize를 이용해서 일정한 크기로 만든다 (256,256)
                    # 4. 잘라낸 optical flow 파일을 list 형식으로 저장한다
                # 5. for문이 종료하고 나서 numpy로 하나의 덩어리로 stack 한다
                # 6. stack 된 파일을 yield 한다
                        concated_list = []
                        for image_index in range(len(json_file_name)):
                            
                            # 이미지 불러오기
                            image_path = arg.image_path+json_file_name[image_index].split('/')[-1][:-4] + 'jpg' # data/for_test/flow/frame0.jpg
                            image = cv2.imread(image_path)
                            # BBox에 맞춰서 자르기
                            h,w,_ = image.shape
                            #print(result[4*image_index+1],result[4*image_index+3], result[4*image_index],result[4*image_index+2])

                            image = image[int(result[4*image_index+1]*h):int(result[4*image_index+3]*h), int(result[4*image_index]*w):int(result[4*image_index+2]*w)] # 잘라진 이미지 -> 좌표 확인 완료
                            image = resize(image, arg.flow_size) # (256,256,3)
                            # 차원 증가
                            image = np.expand_dims(image, axis=0) # (1,256,256,3)
                            print(image_path)
                            print(f'Key :: {key}')
                            #print(result[4*image_index+1],result[4*image_index+3], result[4*image_index],result[4*image_index+2])
                            # optical flow 불러오기
                            #flow_path = arg.flow_path+json_file_name[image_index].split('/')[-1][:-4] + 'flo'
                            flow_path = arg.flow_path+json_file_name[image_index].split('/')[-1][:-4] + 'jpg'
                            flow = cv2.imread(flow_path)
                            h,w,_ = flow.shape
                            #with open(flow_path, 'rb') as f:
                            #    magic = np.fromfile(f, np.float32, count=1)
                            #    w = np.fromfile(f, np.int32, count=1)
                            #    h = np.fromfile(f, np.int32, count=1)
                            #    flow = np.fromfile(f, np.float32, count=2*int(w)*int(h))
                            # optical flow 자르기
                            flow = flow[int(result[4*image_index+1]*h):int(result[4*image_index+3]*h), int(result[4*image_index]*w):int(result[4*image_index+2]*w)]
                            flow = resize(flow, arg.flow_size)
                            # optical flow 차원 증가
                            flow = np.expand_dims(flow,axis=0) # (1,256,256,2)

                            # 이미지와 flow 데이터 합치기
                            image = np.concatenate((image, flow), axis = -1) # (1,256,256,5) 테스트에서는 flow 데이터 대신 이미지 사용해서 (1,256,256,6)
                            concated_list.append(image)

                        series = [i for i in range(arg.model_input)]

                        label_1 = [0,1]
                        label_2 = [0,1]

                        if random.random() < arg.prob:
                            anomaly_choice = random.choice(arg.task_list)
                            if anomaly_choice == 1: # 순서 바꾸기
                                #print('Anomaly Dataset :: Arrow of Time')
                                series.reverse()
                                label_1 = [1,0]
                                label_2 = [1,0]
                                pass 
                            elif anomaly_choice == 2: # motion irregularity
                                #################### 수동으로 설정하기
                                #print('Anomaly Dataset :: Motion Irregularity')
                                temp = random.random()
                                if temp >= 0 and temp >= 0.25:
                                    series[1] = series[0] # t-2, t-2, t ,t+1, t+2
                                    #print('t-2, t-2, t ,t+1, t+2')
                                elif temp >= 0.5:
                                    series[0] = series[1] # t-1, t-1, t, t+1, t+2
                                    #print('t-1, t-1, t, t+1, t+2')
                                elif temp >= 0.75:
                                    series[3] = series[4] # t-2, t-1, t, t+2, t+2
                                    #print('t-2, t-1, t, t+2, t+2')
                                else:
                                    series[4] = series[4] # t-2, t-1, t, t+1, t+1
                                    #print('t-2, t-1, t, t+1, t+1')
                                label_2 = [1,0]
                            
                        concated = None
                        state_ = False
                        for index in series:
                            #print(index)
                            if state_ == False:
                                concated = concated_list[index]
                                state_ = True
                            else:
                                concated = np.concatenate((concated, concated_list[index]), axis = 0)
                            

                        yield concated, result, label_1, label_2, concated_list[mid_index]    
    
    def __new__(cls):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape = (arg.model_input, arg.flow_size[0], arg.flow_size[1], 6), dtype = tf.float32),
                tf.TensorSpec(shape=(20,), dtype=tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.float32),
                tf.TensorSpec(shape=(1, arg.flow_size[0], arg.flow_size[1], 6), dtype=tf.float32),
            )
        )
