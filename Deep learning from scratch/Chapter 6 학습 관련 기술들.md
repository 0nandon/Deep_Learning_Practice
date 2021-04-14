## 매개변수의 갱신
### 1. 확률적 경사 하강법(SGD)
```python
class SGD:
  def __init__(self, lr=0.01):
    self.lr = lr
  
  def update(self, params, grads):
    for key in params.key():
      params[key] -= self.lr * grads[key]
```
위 코드는 SGD를 모듈화하여 코드로 표현한 것이다. SGD의 단점은 비등항성 함수에서는 탐색 경로가 매우 비효율적이라는 것이다.
이러한 SGD의 단점을 개선해주는 것으로, 모멘텀, AdaGrad, Adam 등이 있다.

### 2. 모멘텀
```python
class Momentum:
  def __init__(self, lr=0.01, momentum=0.9):
    self.lr = lr
    self.momentum = momentum
    self.v = None
    
  def update(self, params, grads):
    if self.v is None:
      self.v = {}
      for key, val in params.items():
        self.v[key] = np.zeros_like(val)
    
    for key in params.keys():
      self.v[key] = self.mementum * self.v[key] + self.lr * grads[key]
      grads[key] += self.v[key]
```
모멘텀은 가중치를 수정할 때, 이전의 업데이트 값을 고려해서 수정해 나간다.
### 3. AdaGrad
신경망 학습에서 학습률은 매우 중요하다. 이 값이 너무 작으면 학습 시간이 길어지고, 너무 크면 발산하여 학습이 제대로 이루어지지 않는다.
이 학습률을 정하는 효과적 기술로 학습률 감소(learning rate decay)가 있다. 이는 학습을 진행해 나가면서 점점 학습률을 줄여나가는 방법으로,
이러한 아이디어를 채택한 최적화 방식이 바로 AdaGrad이다.
