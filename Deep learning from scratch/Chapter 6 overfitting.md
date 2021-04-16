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
