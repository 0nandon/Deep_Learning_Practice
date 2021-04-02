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

