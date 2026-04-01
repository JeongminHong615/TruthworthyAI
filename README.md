# Assignment1

neural networks의 adversial attacks의 기법(FGSM, PGD) 구현하고 평가한다. MNIST와 CIFAR-10 데이터셋을 사용하여 공격 수행하였으며 $\epsilon$ 값에 따라 모델의 변화를 시각화한다.

## Features

- 모델
  - MNIST : 구현한 CNN 아키텍처
  - CIFAR-10 : 사전 학습된 ResNet20 모델 사용 (출처 :chenyaofo/pytorch-cifar-models)
- adversial attacks
  - FGSM
  - PGD
- evaluation & visualization
  - $\epsilon$값에 따른 공격 성공률 및 시각화 이미지 자동 저장

## Structure

.
├── test.py # 메인 스크립트
├── requirements.txt # 필요한 파이썬 패키지 목록
├── README.md
├── report.pdf # 실험 결과 분석 보고서
├── mnist_cnn.pth # MNIST CNN 모델 가중치
└── results/ # 시각화 이미지 저장
