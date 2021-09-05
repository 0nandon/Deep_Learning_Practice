## 어텐션에 관한 남은 이야기

### 양방향 RNN

기존의 LSTM은 한방향으로만 시계열 데이터를 처리하였다. 예를들어, '나는 고양이를 좋아한다.' 라는 문장을 처리할 때,
'고양이' 단어의 은닉 벡터에는 '나' 와 '는'에 대한 데이터가 담겨 있을 것이다. 그러나 단어를 처리할 때,
그 단어의 은닉벡터가 단지 앞에 위치한 단어 뿐만 아니라, 주변 단어의 정보를 골고루 반영한다면 문장을 더 정확하게
분석할 수 있을 것이다.

양방향 RNN에서 LSTM은 시계열 데이터를 양방향으로 처리한다. 아래는 LSTM을 양방향으로 처리하는 BiLSTM 클래스를 구현한 것이다.

```python
class TimeBiLSTM:
  def __init__(self, Wx1, Wh1, b1, Wx2, Wh2, b2, stateful=False):
    self.forward_lstm = TimeLSTM(Wx1, Wh1, b1)
    self.backward_lstm = TimeLSTM(Wx2, Wh2, b2)
    self.params = self.forward_lstm.params + self.backward_lstm.params
    self.grads = self.forward_lstm.grads + self.backward_lstm.grads
    
  def forward(self, xs):
    o1 = self.forward_lstm.forward(xs)
    o2 = self.backward_lstm.forward(xs[:, ::-1])
    
    o2 = o2[:, ::-1]
    out = np.concatentate((o1, o2), axis=2)
    return out
    
  def backward(self, dhs):
    H = dhs.shape[2] // 2
    do1 = dhs[:, :, :H]
    do2 = dhs[:, :, H:]
    
    dxs1 = self.forward_lstm.backward(do1)
    dxs2 = self.backward_lstm.backward(do2[:, ::-1])
    dxs2 = dxs2[:, ::-1]
    
    dxs = dxs1 + dxs2
    return dxs
```

### seq2seq 심층화와 skip 연결

번역 등 현실에서의 애플리케이션은 풀어야 할 문제가 훨씬 복잡하다. 즉 지금까지 구현한 seq2seq보다 더 표현력이 높은
모델이 필요하다. 모델의 포현력을 높이는 가장 보편적인 방법은 신경망의 층을 높이는 것이다. 보통은 Encoder와 Decoder를 이루는
LSTM의 층을 동일하게 한다.

층을 깊게 하면 가중치의 개수가 많아져 과대적합이나 기울기 소실 문제 등이 발생한다. RNN의 시간방향으로의 기울기 소실은
**gradient clipping**이나 **LSTM의 여러 게이트** 등으로 어느 정도 방지할 수 있다. 그렇다면 깊이 방향으로의 기울기 소실은 어떻게 방지할까?
보통은 **skip connection** 이라는 기법을 많이 사용한다.

### 어텐션의 응용

#### 구글 신경망 기계 번역 (GNMT)
