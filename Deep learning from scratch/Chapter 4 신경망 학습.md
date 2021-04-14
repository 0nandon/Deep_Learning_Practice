## 4.2 손실함수
신경망은 '하나의 지표'를 기준으로 최적의 매개변수(가중치) 값을 탐색한다. 이 지표를
손실함수(loss function)이라고 한다.

### 4.2.1 오차제곱합
가장 많이 쓰이는 손실 함수는 오차제곱합(sum of squares for error, SSE)이다.
```python
import numpy as np

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)
```
### 4.2.2 교차 엔트로피 오차
그 다음은
교차 엔트로피 오차(cross entropy error, CEE)도 자주 이용한다.

```python
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
```
### 4.2.3 미니배치 학습
지금까지는 데이터 한개 당 손실함수를 한번 계산하였다. 그러나 보통 딥러닝에 사용되는 빅데이터의 개수는
몇 만개, 혹은 몇 억개 까지 가기도 한다. 이 많은 데이터를 대상으로 일일히 손실 함수를 게산하는 것은 현실적이지 않다.
이런 경우 데이터 일부를 추려 전체의 '근사치'로 이용할 수 있다. 신경망 학습에서도 훈련 데이터로부터 일부만 골라
학습을 수행한다. 이것을 미니배치 학습이라고 한다.

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
```
앞에서 얻은 mnist데이터에서 무작위로 100개의 데이터를 골라내겠다.
```python
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_batch[batch_mask]
```
### 4.2.4 (배치용) 교차 엔트로피 오차 구현하기
```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```
위 손실함수는 정답 레이블이 원-핫 인코딩이 아니라 '2'나 '7' 등의 숫자 레이블로 주어졌을 때의
교차 엔트로피 오차는 다음과 같이 구현할 수 있다.
```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```
### 4.2.5 왜 손실 함수를 설정하는가
모델의 성능을 평가하는 지표로는 정확도와 손실함수가 있다. 굳이 손실함수를 사용하는 이유는
경사 하강법을 시행하면서 가중치를 업데이트 할 때, 정확도를 기준으로 미분을 하면, 매개변수의 미분값이
대부분의 지점에서 0이 되기 때문이다. 미분값이 0이 되면 학습이 안된다는 의미이므로 손실함수를 사용한다.

같은 맥락에서, 활성화 함수로 시그모이드를 자주 이용하는 이유는, 시그모이드는 미분값 0인 지점이 없기 때문이다.

### 4.4.2 신경망에서의 기울기
x0와 x1 편미분을 한번에 계산하는 코드를 구현한다.
```python
import numpy as np

def numerical_gradient():
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) * 2h
        x[idx] = tmp_val
     
     return grad
```
위 함수를 사용해서 실제로 구하기를 구하는 코드를 구현해 보겠다.
```python
import sys, os
sys.path.append(os.pardir)
import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버 플로우를 방지하기 위해 c를 빼준다.
    return exp_a / np.sum(exp_a)

def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-7))

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
        
    def predict(self, x):
        a = np.dot(x, self.W)
        z = np.softmax(a)
        
    def loss(self, x, t):
        z = predict(x)
        loss = np.cross_entropy_error(z, t)
        return loss
```
