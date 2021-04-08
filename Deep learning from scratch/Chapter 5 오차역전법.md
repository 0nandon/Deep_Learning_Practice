## 오차 역전파법
Chapter4에서는 수치미분으로 학습을 구현하였다. 수치 미분은 단순하고 구현하기 쉽지만, 계산 시간이 오래 걸린다는
단점이 있다. 이번 장에서는 가중치 매개변수의 기울기를 효율적으로 계산하는 '오차역전파법'(back propagation)을 배워보겠다.

### 5.4 단순한 계층 구현하기
#### 5.4.1 곱셈 계층

```python
class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None
  
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x * y
    return out
   
  def backward(self, dout):
    dx = dout * self.y
    dy = dout * self.x
    
    return dx, dy
```

#### 5.4.2 덧셉 계층
```python
class AddLayer:
  def __init__(self):
    pass  # 덧셉 과정은 초기화가 필요없다.
  
  def forward(self, x, y):
    out = x + y
    return out
  
  def backward(self, dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy
```
### 5.5 활성화 함수 계층 구현하기
#### 5.5.1 ReLU 계층
```python
class ReLU:
  def __init__(self):
    self.mask = None
   
  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0
    return out
  
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout
    return dx
```
#### 5.5.2 Sigmoid 계층
```python 
class Sigmoid:
  def __init__(self):
    self.out = None
  
  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out
    return out
  
  def backward(self, dout):
    dx = dout * (1.0 - self.out) * self.out
    return dx

```
### 5.6 Affine/Softmax 계층 구현하기
#### 5.6.1 Affine 계층
```python
class Affine:
  def __init__(self):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None
  
  def forward(self, x):
    self.x = x
    a = np.dot(x, W) + b
    return a
   
  def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = dout
    
    return dx
```
#### 5.6.2 Softmax-with-Loss 계층
