Tensorflow Enet
===============

[[개요]](#개요)

[[Enet 사용 환경 셋업]](#enet-사용-환경-셋업)

[[enet 다운로드]](#enet-다운로드)

[[tensorflow 설치]](#tensorflow-설치jetson-tx2-board)

[[cudnn & CUDA]](#cudnn-cuda)

[[Enet 실행하기]](#enet-실행하기)

[[Tensorflow-enet]](#tensorflow-enet)

[[학습]](#학습)

[[학습된 모델 이용]](#학습된-모델-이용)

개요
====

-   Tensorflow Enet을 사용하기 위한 환경설정 방법과 사용법을 설명한다.

-   Enet은 Real-Time Semantic Segmentation을 가능하게 하는 딥 러닝
    네트워크 모델이다.

-   [[ENet: A Deep Neural Network Architecture for Real-Time Semantic
    Segmentation]](https://arxiv.org/pdf/1606.02147.pdf)

![](https://user-images.githubusercontent.com/44761194/49774471-3987d800-fd38-11e8-87a0-6ebdb7a3465a.png)

![](https://user-images.githubusercontent.com/44761194/49774477-3d1b5f00-fd38-11e8-8ea6-9050d700764b.png)

ENET은 임베디드 보드에서 실시간 영상 처리를 위해 속도를 대폭적으로 향상 시킨 딥러닝 네트워크 모델이다. NVIDIA TX1 임베디드 보드에서 480x320 픽셀 기준 21.1fps 의 영상 처리 속도를 낼 수 있으며 이는 수많은 논문에서의 실시간 기준인 20\~30fps의 범위에 속한다.

Enet 사용 환경 셋업 
====================

**enet 다운로드**
-----------------

-   [[https://github.com/kwotsin/TensorFlow-ENet]](https://github.com/kwotsin/TensorFlow-ENet)

-   git clone https://github.com/kwotsin/TensorFlow-ENet.git

-  eNet을 다운받은 소스코드 위치에서 https://drive.google.com/file/d/1NGg118IhjK1kXGdDW7TDHLfFoPfogDhe/view?usp=sharing 를 
   압축을 해제한다.

tensorflow 설치(Jetson tx2 board)
---------------------------------

-   Enet **Requirements:** TensorFlow = r1.2

-   Tensorflow-gpu 설치는 다음의 Nvidia 가이드 라인을 따른다.(1.2 버전을 설치해야 한다.)
    -   [[https://docs.nvidia.com/deeplearning/dgx/install-tf-jetsontx2/index.html]](https://docs.nvidia.com/deeplearning/dgx/install-tf-jetsontx2/index.html)

**cudnn & CUDA**
----------------

-   Tensorflow-gpu를 사용하기 위해서는 CUDNN과 CUDA를 설치해야 한다.

-   Tensorflow-gpu 1.2버전에 맞는 CUDNN 5.1, CUDA8 버전을 Nvidia에서 다운로드 하여 설치한다.

> ![](https://user-images.githubusercontent.com/44761194/49774493-4ad0e480-fd38-11e8-98b6-3dc9b4c99e9c.png)

Enet 실행하기
=============

Tensorflow-enet
---------------

![](https://user-images.githubusercontent.com/44761194/49774498-4e646b80-fd38-11e8-9c86-dfe42084024e.png)
-   Tensorflow-Enet을 다운받으면 다음과 같은 파일들이 있다.
-   Checkpoint 폴더 : 학습된 모델이 들어있는 폴더
-   Dataset 폴더 : 학습데이터셋이 들어있는 폴더
-   Visualizations 폴더 : gif 형태의 output이 들어있는 폴더
-   Enet.py : Enet이 구현되어있는 파일
-   Get\_class\_weights : class들의 가중치를 구하는 파일
-   Predict\_segmentation.py : 단위테스트를 위한 파일
-   Preprocessing.py : 입력 사진 전처리를 위한 파일
-   Test\_enet.py : 새로운 이미지에 대해 학습된 가중치를 이용해 처리하여 결과 이미지를 얻는 파일
-   Test.sh : test\_enet.py 파일을 이용하는 예시 스크립트 파일
-   Train\_enet.py : 학습 dataset을 이용해 Enet을 학습시키는 파일
-   Train.sh : train\_enet.py 파일을 이용하는 예시 스크립트 파일

학습
----

-   Train\_enet.py 파일을 이용한다.

    -   명령어 : python train\_enet.py
    -   Train.sh 에 있는 예시 코드를 참고
        ![](https://user-images.githubusercontent.com/44761194/49774503-53291f80-fd38-11e8-8846-66fa00270595.png)
    -   학습시 dataset 폴더에 있는 학습 데이터를 이용한다.
    -   Train 폴더 : 학습 데이터(original 이미지)
    -   Trainannot 폴더 : 학습 데이터에 대한 라벨링 데이터(Label 이미지)
    -   Val 폴더 : 학습된 모델을 평가하기위한 데이터(original 이미지)
    -   Valannot 폴더 : val 폴더의 데이터에 대한 라벨링 데이터(Label 이미지)
    -   Test 폴더와 testannot 폴더는 이용하지 않는다. (단위 테스트를 위해 임시로 만들어져 있는 폴더)

학습된 모델 이용
----------------

-   Test\_enet.py 파일을 이용한다

    -   명령어 : python test\_enet.py

    -   Test.sh 에 있는 예시 코드를 참고

    -   test 결과로 다음과 같은 결과물들을 얻을 수 있다.(원본 이미지와 테스팅 결과 이미지)
![image](https://user-images.githubusercontent.com/44758482/49780628-b58f1980-fd52-11e8-85a4-037c4aafa087.png)
![image](https://user-images.githubusercontent.com/44758482/49780617-a740fd80-fd52-11e8-9dde-3dc2d1eaf8f7.png)

