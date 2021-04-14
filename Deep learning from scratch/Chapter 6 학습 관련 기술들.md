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

