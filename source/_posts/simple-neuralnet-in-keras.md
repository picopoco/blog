title: Keras로 개발하는 뉴럴넷
date: 2017-01-26 12:00:00
---

텐서플로우가 핫하다지만 처음 개발자가 접하기엔 어려운 부분이 있을 수 있습니다. 케라스는 텐서플로우의 많은 기능을 추상화 시켜서 코드를 더 짧게 만들어주며 쉽게 배울 수 있는 장점이 있어 비전공자나 처음 접하시는 분들이 사용하기에 좋습니다.

<!--more-->

<div class="tip">
  **주의사항**: 체계를 갖추고 쓴 글이 아니기 때문에 글쓴이의 의식의 흐름에 좀 더 가깝습니다. 텐서플로우나 다른 프레임워크로 한번 코딩을 해보신들에게 케라스를 소개하기 위해 작성된 글입니다..
</div>

## Keras는 무엇인가?

Keras는 파이썬 딥러닝 프레임워크의 양대 산맥인 Tensorflow와 Theano를 감싸서 더 간결하고 쉬운 API를 제공하는 뉴럴넷 라이브러리 입니다. 코드를 최소화 시키고 직관적이면서도 확장성 있는 디자인에 초점을 맞추고 계획된 라이브러리죠. 딥러닝을 처음 시작할때, 그리고 프로토타이핑을 할때 사용하기 좋으며 모듈화를 통한 확장성도 좋아서 여러 프로젝트에 쓰이고 있습니다. 딥러닝을 처음 접할때도 좋지만 규모가 큰 프로젝트에도 쓰일 수 있을 정도로 안정성과 성능이 보장되는 편입니다.

## Keras의 특징

기본적인 기능들은 Layers와 그들의 input과 output으로 이루어져있습니다. 개별 뉴런에는 접근할 수 없고 모든것이 층(Layer) 으로 구현됩니다. Keras는 편의를 위해 개발자들이 자주 쓰는 layer들을 미리 제공하고 있습니다. 기본적인 dense한 여러겹의 뉴럴넷(MLP)은 아래와 같이 구현됩니다. (파라미터들은 일단 무시 하겠습니다):

```python
keras.layers.core.Dense(output_dim, activation='linear')
```

다른 예로 자연어 처리에 많이 쓰이는 RNN, LSTM, GRU layer등은 아래와 같이 구현됩니다.

```python
keras.layers.recurrent.GRU(output_dim, ...) # 생략
```

이와 같이 Keras는 한줄의 코드로 뉴럴넷의 Layer를 구현 가능하게 해줍니다.

## 튜토리얼 개요

아래의 카테고리별로 케라스를 어떻게 사용하는지 차근차근 알아보겠습니다.

1. 데이터 로딩
2. 모델 정의하기
3. 모델 컴파일하기
4. 모델 Fit하기
5. 모델 평가
6. 모두다 엮어서 써보기

컴파일? Fit? 무슨말인지 감이 안올 수 있습니다. 아래에 설명을 이어서 할테니 계속 알아보겠습니다.

## 1 데이터 로딩

무작위의 숫자를 뽑는과정을 거치는 모든 머신러닝 알고리즘을 개발할땐 random seed를 지정해주는 것이 좋습니다. 랜덤시드는 같은 코드를 여러번 돌렸을때 같은 결과가 나오게 해줍니다. 아래와 같이 개발할때는 케라스를 import 하면서 랜덤시드도 지정해주는 습관을 들이면 디버깅할때 스트레스를 덜 받게 됩니다.

```python
from keras.models import Sequential
from keras.layers import Dense
import numpy

seed = 7
numpy.random.seed(seed)
```

Sequential모델과 Dense층을 케라스로부터 가져왔는데 나중에 설명하겠습니다. 이제 데이터를 로딩해보죠.

```python
dataset = numpy.loadtxt('dataset.csv', delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]
```

## 2 모델 정의하기

케라스에서 Sequential 모델은 층(Layer)를 기본 단위로 그 층들이 순서대로 쌓이며 만들어집니다. 개발자는 레고처럼 알맞은 층을 찾아서 레고처럼 쌓아주기만 하면 됩니다. 아까 케라스에서 가져온 Sequential 모델에 이것저것 붙여가면서 만들어보겠습니다.

주의할점은 첫번째 층은 `input_dim` 을 인풋의 갯수 (피쳐 수)로 맞춰워야 한다는 것입니다. 예를들어 데이터가 5열로 이루어진 csv데이터면 `input_dim`을 5로 지정해주면 됩니다.

*층의 갯수와 종류는 어떻게 결정하는가?*라고 궁금증을 가질 수 있습니다. 사실 층의 갯수가 얼마나 필요한지는 아무도 모릅니다. 경험으로 얻어진다고는 하는데 잘 모르겠습니다. 일단 그럴듯 해보이는 숫자를 넣어봅시다.

아까 나왔던 Dense는 Fully Connected Layers 라고도 하는 가장 기본적인 레이어 입니다. 이 뉴럴넷 층을 불러오면서 뉴론의 갯수, 초기화 방법, activation function을 지정해줄 수 있습니다.

```python
# Sequential Model을 인스턴스화 해서 층들이 쌓일 기초를 마련
model = Sequential()

# 각 층은 Sequential 모델 안의 add 함수로 더할 수 있다.

# 첫 층은 12개의 뉴론을 가지며 8 가지의 인풋을 받는다.
# 뉴런의 weight 값은 uniform distribution으로 초기화 되며 엑티베이션 함수는 'relu'를 사용한다.
model.add(Dense(12, input_dim=5, init='uniform', activation='relu'))

# 보통 Hidden layer라고 불리는 두번째 층은 8개의 뉴런을 갖는다.
model.add(Dense(8, init='uniform', activation='relu'))

# 마지막으로 아웃풋 레이어는 binary classification이기 때문에 1개의 뉴런을 갖는다.
model.add(Dense(1, init='uniform', activation='sigmoid'))
```

정말 쉽게 뉴럴넷을 구현하였습니다. 역전파(Backpropagation)는 다른 프레임워크들 처럼 자동으로 계산을 해주기 때문에 우리는 앞으로 진행되는 코드만 짜면 됩니다.

## 3 모델 컴파일하기

모델을 사용하기 위해선 정의 후 컴파일이라는 단계를 거쳐야 합니다. 해보겠습니다.

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

여기선 이 한줄이 전부입니다. 간단해 보이지만 이 코드를 입력하는 순간 복잡한 연산들이 일어나게 됩니다. 사용자가 지정한 backend (여기선 Tensorflow)를 이용하여 이 모델을 어떻게 가장 효율적으로 나타낼 수 있는지, CPU나 GPU를 어떻게 사용하는지, 이 과정에서 모두 결정이 됩니다. 사용자는 loss 계산법, optimizer등을 파라미터로 넘겨주기만 하면 됩니다. 텐서플로우에선 여기서 넘겨주는 파라미터 각각 한줄 혹은 여러줄을 차지하는데, 케라스는 한줄로 모든걸 해결해줍니다.

## 4 모델 Fit

sklearn이라는 라이브러리를 사용했으면 익숙할 수 있는 과정입니다. Fit은 우리가 정의하고 컴파일한 모델에 데이터를 먹이는 과정입니다. 아까 데이터를 `X`와 `Y`에 담았었었다. 모델에 먹여보겠습니다.

```python
model.fit(X,Y, nb_epoch=150, batch_size=10)
```

## 5 모델 평가

모델을 훈련시키고 이제 우리가 정성스럽게 만든 모델을 어떻게 평가하고 점수를 줄 것인가 지정을 해주겠습니다. 모델이 훈련할때 보지 못했던 데이터로 예측을 어떻게 하는지 측정해보겠습니다. 테스트 데이터셋을 따로 준비해준다음 아까 Fit할때 했던것처럼 모델에 먹여주기만 하면 됩니다.

```python
scores = model.evaulate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

## 6 합쳐보기

뭔가 설명을 많이 한 것 같지만 이 모든것이 16 줄에 담깁니다.

```python
from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)
dataset = numpy.loadtxt("dataset.csv", delimiter=",")
testset = numpy.loadtxt("testset.csv", delimiter=",")
X, Y = dataset[:,0:5], dataset[:,5]
X_test, Y_test = testset[:,0:5], testset[:,5]
model = Sequential()
model.add(Dense(12, input_dim=5, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, nb_epoch=150, batch_size=10)
scores = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

돌려보면 아래와 같은 결과가 나오게 됩니다.


```python
...
Epoch 143/150
768/768 [==============================] - 0s - loss: 0.4614 - acc: 0.7878
Epoch 144/150
768/768 [==============================] - 0s - loss: 0.4508 - acc: 0.7969
Epoch 145/150
768/768 [==============================] - 0s - loss: 0.4580 - acc: 0.7747
Epoch 146/150
768/768 [==============================] - 0s - loss: 0.4627 - acc: 0.7812
Epoch 147/150
768/768 [==============================] - 0s - loss: 0.4531 - acc: 0.7943
Epoch 148/150
768/768 [==============================] - 0s - loss: 0.4656 - acc: 0.7734
Epoch 149/150
768/768 [==============================] - 0s - loss: 0.4566 - acc: 0.7839
Epoch 150/150
768/768 [==============================] - 0s - loss: 0.4593 - acc: 0.7839
768/768 [==============================] - 0s
acc: 79.56%
```


