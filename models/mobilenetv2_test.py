import torch
import torch.nn as nn
from torchvision import models, transforms, datasets

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 테스트 데이터셋 로드
test_dataset = datasets.ImageFolder(root=r'C:\Microsoft\OneDrive\바탕 화면\dataset\test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# MobileNetV2 모델 초기화
model = models.mobilenet_v2(pretrained=False, num_classes=3)

# 모델 경로 설정
model_path = r'C:\Users\airyt\Downloads/trained_model_final.pt'

# 저장된 가중치 불러오기 (state_dict만 불러오기)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# 평가 모드로 설정
model.eval()

# 평가 수행
correct = 0  # 정확하게 예측한 수 초기화
total = 0  # 전체 샘플 수 초기화
loss_function = torch.nn.CrossEntropyLoss()  # 손실 함수 정의

# 그래디언트 계산을 비활성화하여 성능 평가
with torch.no_grad():
    # 테스트 데이터셋의 각 배치에 대해서 반복
    for inputs, labels in test_loader:
        # 모델에 이미지 전달하여 예측값 계산
        outputs = model(inputs)

        # 가장 높은 확률을 가진 클래스 선택
        _, predicted = torch.max(outputs.data, 1)

        # 정확하게 예측한 수 및 전체 샘플 수 업데이트
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 손실 계산
        loss = loss_function(outputs, labels)

'''
전체 정확도 및 손실
'''
# 정확도 계산 및 출력
accuracy = 100 * correct / total
print(f'테스트 정확도: {accuracy:.2f}%')

# 손실 출력
print(f'테스트 손실: {loss:.4f}')

'''
클래스별 정확도 및 손실
'''
# 클래스 이름을 담은 리스트 정의
class_names = ['컵', '맥주잔', '소주잔']

# 각 클래스별로 정확한 예측 수를 저장하는 리스트 초기화
class_correct = [0. for _ in range(3)]

# 각 클래스별 전체 샘플 수를 저장하는 리스트 초기화
class_total = [0. for _ in range(3)]

# 각 클래스별로 손실 값을 저장하는 리스트 초기화
class_losses = [0. for _ in range(3)]

# 그래디언트 계산을 비활성화하여 성능 평가를 위한 코드 부분
with torch.no_grad():
    # 테스트 데이터셋의 각 배치에 대해서 반복
    for inputs, labels in test_loader:
        # 모델에 이미지 전달하여 예측값 계산
        outputs = model(inputs)

        # 손실 계산
        loss = loss_function(outputs, labels)

        # 예측된 클래스 중 가장 높은 값 선택
        _, predicted = torch.max(outputs, 1)

        # 정확한 예측 여부 확인 및 클래스별로 업데이트
        for i in range(len(labels)):
            label = labels[i]
            class_losses[label] += loss.item()
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

# 각 클래스별로 정확도와 손실 출력
for i in range(3):
    print(f'클래스 {i} ({class_names[i]})의 정확도: {100 * class_correct[i] / class_total[i]:.2f}%')
    print(f'클래스 {i} ({class_names[i]})의 평균 손실: {class_losses[i] / class_total[i]:.4f}')
