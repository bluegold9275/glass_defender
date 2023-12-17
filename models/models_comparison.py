import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
import time
import psutil
import os

# 클래스와 데이터 수
num_classes = 3
data_size = 600  # 한 클래스 당 200개로 가정

# 임의의 이미지 크기 (224x224)
input_shape = (224, 224, 3)

# 모델 불러오기
# 전이학습 : 최상위 레이어 사용 X = 네트워크의 특성 추출기 부분만 사용
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
resnet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
mobilenetv2_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

# 데이터 생성 (임의의 데이터)
dummy_data = tf.random.normal((data_size, *input_shape))

# 각 모델의 추론 속도와 리소스 측정
def measure_inference_speed(model, data):
    start_time = time.time()
    _ = model.predict(data)
    end_time = time.time()
    return end_time - start_time

def measure_resource_usage(model):
    pid = os.getpid()  # 현재 프로세스의 ID
    py = psutil.Process(pid)  # 현재 프로세스의 정보를 가져오기 위해 psutil.Process 객체를 생성
    memory_usage = py.memory_info()[0] / (2 ** 30)  # 현재 프로세스의 메모리 사용량을 가져와서 GB 단위로 변환
    return memory_usage  # 계산된 메모리 사용량을 반환

# 각 모델의 추론 속도 측정
vgg16_speed = measure_inference_speed(vgg16_model, dummy_data)
resnet50_speed = measure_inference_speed(resnet50_model, dummy_data)
mobilenetv2_speed = measure_inference_speed(mobilenetv2_model, dummy_data)

# 각 모델의 메모리 사용량 측정
vgg16_memory = measure_resource_usage(vgg16_model)
resnet50_memory = measure_resource_usage(resnet50_model)
mobilenetv2_memory = measure_resource_usage(mobilenetv2_model)

# 결과 출력
print(f"VGG-16 추론 속도: {vgg16_speed} 초")
print(f"VGG-16 리소스 사용량: {vgg16_memory} GB\n")
print(f"ResNet-50 추론 속도: {resnet50_speed} 초")
print(f"ResNet-50 리소스 사용량: {resnet50_memory} GB\n")
print(f"MobileNetV2 추론 속도: {mobilenetv2_speed} 초")
print(f"MobileNetV2 리소스 사용량: {mobilenetv2_memory} GB")