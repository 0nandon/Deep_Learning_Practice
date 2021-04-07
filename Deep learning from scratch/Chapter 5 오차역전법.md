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
    self.x = None
    slef.y = None
  
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x + y
    return out
  
  def backward(self, dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy
```
