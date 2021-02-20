
## 다중 입력 모델
keras는 함수형 API(functional api)를 통해 다중 입력 모델을 구현할 수 있다. 아래와 같은 함수들이 사용된다.
* keras.layers.add
* keras.layers.concatenate

아래는 예시코드이다.
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, concatenate, Dense, Activation

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape = (None,), dtype='int32', name='text')
embedded_text = Embedding(question_vocabulary_size, 64)(text_input)
encoded_text = LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32', name='text')
embedded_question = Embedding(question_input, 32)(question_input)
encoded_question = LSTM(16)(question_input)

concatenated = concatenate([encoded_text, encoded_question])
answer = Dense(1)(answer_vocabulary_size)(concatenate)
answer = Activation('softmax')(answer)

model = Model(inputs = [text_input, quesiton_input], outputs = answer)
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['acc'])
```

## 다중 출력 모델
다중 출력 모델 또한 구현할 수 있다. 보통 출력 마다 다른 손실 함수를 지정합니다. 손실 함수를 지정하면, 함수에 따라 결과값의
스케일이 달라지는데, 이러한 스케일을 맞추기 위해 각 손실 함수 값에 가중치를 부여한다. 예를들어 MSE(평균 제곱 오차)는 평균적으로 3 ~ 5 사이의 값을 가지는 반면,
크로스 엔트로피 손실은 0.1로 그 값이 매우 낮다. 딥러닝 모델은 손실 값이 큰 작업에 치우쳐 학습을 하기 때문에 

아래는 예시 코드이다.
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Conv1D, Maxpooling1D, GlobalMaxPooling1D

vocabulary_size = 50000
num_income_groups = 0

posts_input = Input(shape = (None,), dtype='int32', name='posts')
embedded_posts = Embedding(vocabulary_size, 256)(posts_inuts)
x = Conv1D(128, 5, activation='relu')(embedded_posts)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = Conv1D(256, 5, activation='relu')(x)
x = Maxpooling1D(5)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = Conv1D(256, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)

# 다중 출력
age_prediction = Dense(1, name='age')(x)
income_prediction = Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_predictions, income_predictions, gender_predictions])
model.compile(optimizer='rmsprop',
              loss = ['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights = [0.25, 1., 10.]) # 각 손실함수 값에 가중치를 부여한다.
```

## 층으로 구성된 비순환 유향 그래프
케라스는 모든 비순환 유향 그래프(directed acyclic graph) 형태로 된 모델을 구현할 수 있다.
그러나 사이클을 가지는 유향 그래프 형태의 모델은 구현할 수 없다. 그래프 형태의 모델 중 가장 유명한 2개가
인셉션 모듈(Inception)과 잔차 연결(residual connection)이다. 

### 인셉션 모듈
 인셉션은 합성곱 신경망에서 인기 있는 네트워크 구조이다. 나란히 분리된 가지를 따라 모듈을 쌓아,
 독립된 작은 네트워크처럼 구성한다. 여러 형태의 합성곱으로 이루어진 가지들이 최종적으로 합쳐지며 구성된다.
 이러한 구성은 네트워크가 따로따로 공간 특성과 채널 방향의 특성을 학습하도록 돕는다.
 
 > **Note** <br>
 > 인셉션 모듈에서는 1 * 1 합성곱이 자주 사용된다. 이 합성곱 연산은 한번에 하나의 타일만 처리하기 때문에,
 > 공간 방향으로는 정보를 섞제 않는다.(spatial relation을 고려하지 않는다.) 따라서, 채널 방향의 특성 학습과 공간 방향의 특성학습을 분리하는데
 > 도움을 준다. 또한, 이 합성곱은 특성 맵의 채널 수를 줄여서(dimension reduction), 가중치의 개수를 줄이는 목적으로도 자주 사용된다.
 > 특성 맵의 채널 수보다, 1 * 1 합성곱 필터의 수를 줄이면 dimension reduction 효과를 줄 수 있다. 이러한 형태를 bottle neck이라고 부른다.
 
 아래는 예시 코드이다.
 ```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, AveragePooling2D

# 여기서 x는 4D 텐서라고 하자.
branch_a = Conv2D(128, 1, activation='relu', strides=2)(x)

branch_b = Conv2D(128, 1, activation='relu')(x)
branch_b = Conv2D(128, 3, activation='relu', strides=2)(branch_b)

branch_c = AveragePooling2D(3, strides=2)(x)
branch_c = Conv2D(128, 3, activation='relu')(branch_c)

branch_d = Conv2D(128, 1, activation='relu')(x) # 1*1 합성곱
branch_d = Conv2D(128, 3, activation='relu')(branch_d)
branch_d = Conv2D(128, 3, activation='relu', strides=2)(branch_d)

output = concatenate([branch_a, branch_b, branch_c, branch_d])
 ```
 
 ### 잔차 연결(residual connection)
 잔차 연결은 대규모 딥러닝 모델에서 흔히 나타나는 두 가지 문제인 그래디언트 소실 문제(Gradient vanishing problem)과
 표현 병목(representational bottleneck)을 해결했다. 일반적으로 10개 층 이상을 가진 모델에 잔차 연결을 추가하면
 도움이 된다. 잔차 연결은 하위층의 출력이 상위층의 출력에 더해지는 형태로 구현된다.
 
 아래는 예시 코드이다.
 ```python
 # ...
 y = Conv2D(128, 3, activation='relu', padding='same')(x)
 y = Conv2D(128, 3, activation='relu', padding='same')(y)
 y = Conv2D(128, 3, activation='relu', padding='same')(y)
 
 y = add([y, x]) # 하위층의 출력과 상위층의 출력을 더한다.
 # ...
 ```
 
 > **Note 딥러닝의 표현 병목** <br>
 > 표현 병목 현상은 CNN에서 풀링을 할 때, 특성맵의 크기가 작아지면서 정보가 손실되는 현상을 말한다.
 
 > **Note 그레디언트 소실 문제** <br>
 > 심층 신경망을 훈련하는 데 사용되는 핵심 알고리즘인 역전파는 출력 손실에서 얻은 피드백 신호를 하위 층에
 > 전파한다. 피드백 신호가 깊이 쌓인 층을 통과하여 전파되면 신호가 아주 작아지거나 완전히 사라질 수도 있다.
 > 이렇게 되면 네트워크가 훈련되지 않는다. 이러한 문제를 Gradient vanishing problem 이라고 한다. RNN에서는
 > LSTM으로 이러한 문제를 해결하였고, 다른 신경망에서는 잔차연결로 이러한 문제를 보완한다.
