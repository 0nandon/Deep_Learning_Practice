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
