import tensorflow as tf
import numpy as np
import train_argument as arg
import pandas as pd
import json
# 해당 클래스는 데이터 제너레이터
# fit 함수 안에 클래스로 넣으면 작동할 듯
class ano_dataset(tf.data.Dataset):
    def __init__(self):
        print('Hello World')
        
        print(self.df.type)
        self.length = len(self.df)

    def load_json(self, NAME):
        print('Shuffle Data Frame')
        margin = (arg.model_input-1)//2
        #result = [0 for _ in range(arg.object_count*arg.model_input)] # 반환하기 위한 빈 리스트
        result = []

        for i in range(arg.model_input):
            temp_result = []
            json_path = f'{arg.json_path+str(int(NAME)-margin+i)}.json' # 열 json 파일의 주소
            print(f'Opened Json :: {str(int(NAME)-margin+i)}')
            with open(json_path, 'r') as f:
                data = json.load(f)
            for i in range(len(data['xmin'])):
                temp_result.append(int(data['xmin'][str(i)]))
                temp_result.append(int(data['ymin'][str(i)]))
                temp_result.append(int(data['xmax'][str(i)]))
                temp_result.append(int(data['ymax'][str(i)]))
                if len(temp_result) >= arg.object_count: # 10개의 오브젝트만 확인한다
                    break
            while len(temp_result) != arg.object_count: # 10개의 오브젝트보다 작다면 0을 추가한다
                temp_result.append(0)
            result = result + temp_result
            
        return result

    def _generator(self):
        def load_json( NAME):
            
            margin = (arg.model_input-1)//2
            #result = [0 for _ in range(arg.object_count*arg.model_input)] # 반환하기 위한 빈 리스트
            result = []

            for i in range(arg.model_input):
                temp_result = []
                json_path = f'{arg.json_path+str(int(NAME)-margin+i)}.json' # 열 json 파일의 주소
                print(f'Opened Json :: {str(int(NAME)-margin+i)}')
                with open(json_path, 'r') as f:
                    data = json.load(f)
                for i in (data['xmin'].keys()):
                    temp_result.append(int(data['xmin'][str(i)]))
                    temp_result.append(int(data['ymin'][str(i)]))
                    temp_result.append(int(data['xmax'][str(i)]))
                    temp_result.append(int(data['ymax'][str(i)]))
                    if len(temp_result) >= arg.object_count: # 10개의 오브젝트만 확인한다
                        break
                while len(temp_result) != arg.object_count: # 10개의 오브젝트보다 작다면 0을 추가한다
                    temp_result.append(0)
                result = result + temp_result
                
            return result

        df = pd.read_csv(arg.data_csv_path)
        df = df.sample(frac=1) # 데이터 프레임 셔플
        for _, row in df.iterrows():
            _, name = row['Path'].split('/')
            print(f'Middle Name :: {name}')
            result = load_json(name[:-4]) # bbox정보가 합쳐져서 들어온다. list 형식으로 길이는 40*5. 과거부터 들어온다
            print('')
            yield (result)
    
    def __new__(cls):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=tf.TensorSpec(shape = arg.object_count * arg.model_input, dtype = tf.int64),
            args = (arg.num_samples, )
        )

if __name__ == '__main__':
    dataset = ano_dataset()
    for sample in dataset:
        print(sample.shape)