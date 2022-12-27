import train_argument as arg
import tensorflow as tf
import numpy as np
import cv2 as cv
from sklearn.metrics import confusion
import os
import model_four_tasks as m

### 학습 시작 전 필요 변수
### 초기 변수 바꾸기 위해서는 train_argument.py 내의 변수 바꾸기
class trainer:
    """
    default model_name == model_deep_wide
    """
    def __init__(self, model_name = 'model_deep_wide'):
        self.input_size = arg.input_size # 모델의 입력 크기. Default = [64,64,3]
        self.learning_rate_init = arg.learning_rate_init # 모델 학습률. Default = 10 ^ -3
        self.num_epochs = arg.num_epochs # 학습 epoch. Default = 100
        self.batch_size = arg.batch_size # Batch size. Default = 64
        self.checkpoint_foler = os.path.join # 저장 위치 선정. result/YYYY_MM_DD_HH_MM_SS_lr_{lr}_ep_{epoch}_batch_{batch}

        # 모델 초기 구성
        if model_name == 'model':
            self.model = m.model()
        elif model_name == 'model_wide':
            self.model = m.model_wide()
        elif model_name == 'model_deep':
            self.model = m.model_deep
        elif model_name == 'model_deep_wide':
            self.model = m.model_deep_wide
        else:
            raise NameError('Check model name')
        
        self.global_step = 0 # 뭐할때 쓰는거지?
        self.learning_rate = self.learning_rate_init # 학습률. 나중에 바뀔 수 있어서 이렇게 해놓았나?

        # cost, avg_cose, decoder_middle_ 등은 train, evaluate 결과로 나옴 -> tensorflow2 에서는 필요없을 듯 함


    def run(self, data_loader):