## 오버피팅
오버피팅은 주로 다음의 두 경우에 일어난다.
* 매개변수가 많은 표현력이 높은 모델
* 훈련 데이터가 적용

### 가중치 규제 (l2 norm)
가중치 규제는 손실함수에 노름을 더하여, 가중치를 감소(weight decay)를 시키는 방식이다.

노름에는 L1, L2, L∞ 이 있다.
* L1
```python
def reg_loss(loss, l1=0.1):
  return loss + l1 * np.sum(np.abs(W)) # 손실함수에 l1 노름을 더한다.
```
* L2
```python
def reg_loss(loss, l2=0.1):
  return loss + (l2 / 2) * np.sum(W ** 2) # 손실함수에 l2 노름을 더한다. 
```
* L∞
```python
def reg_loss(loss, l_max=0.1):
  return loss + l_max * np.maximum(np.abs(W)) # 손실함수에 Max 노름을 더한다.
```

보통 L1, L2 노름을 같이 사용한다.
```python
def reg_loss(loss, l1=0.1, l2=0.1):
  return loss + l1 * np.sum(np.abs(W)) + (l2 / 2) * np.sum(W ** 2)
```

### 드롭아웃
가중치 규제는 효과가 있지만, 모델이 복잡해 지면 규제만으로는 부족하다. 이럴때는, 흔히 드롭아웃이라는 기법을 사용한다.
```python
class Dropout:
  def __init__(self, dropout_ratio=0.5):
    self.dropout_ratio = dropout_ratio
    self.mask = None
    
  def forward(self, x, train_flg=True):
    if trian_flg:
      self.mask = np.random.rand(x.shape) > self.dropout_ratio
      return x * self.mask
    else:
      return x * (1.0 - self.dropout_ratio)
  
  def backward(self, dout):
    return dout * self.mask
```
