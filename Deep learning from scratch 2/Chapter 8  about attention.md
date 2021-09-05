## 어텐션에 관한 남은 이야기

### 1. 양방향 RNN

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

### 2. seq2seq 심층화와 skip 연결

번역 등 현실에서의 애플리케이션은 풀어야 할 문제가 훨씬 복잡하다. 즉 지금까지 구현한 seq2seq보다 더 표현력이 높은
모델이 필요하다. 모델의 포현력을 높이는 가장 보편적인 방법은 신경망의 층을 높이는 것이다. 보통은 Encoder와 Decoder를 이루는
LSTM의 층을 동일하게 한다.

층을 깊게 하면 가중치의 개수가 많아져 과대적합이나 기울기 소실 문제 등이 발생한다. RNN의 시간방향으로의 기울기 소실은
**gradient clipping**이나 **LSTM의 여러 게이트** 등으로 어느 정도 방지할 수 있다. 그렇다면 깊이 방향으로의 기울기 소실은 어떻게 방지할까?
보통은 **skip connection** 이라는 기법을 많이 사용한다.

### 3. 어텐션의 응용

#### 3-1. 구글 신경망 기계 번역 (GNMT)

GNMT는 구글번역기에서 사용하는 번역 신경망으로, LSTM 계층의 다층화, 양방향 LSTM, skip connection 등의 기법을 활용하여,
사람과 거의 유사할 정도의 번역 능력을 구현하는 신경망을 구축하였다. 하지만, 아직은 자연스럽지 못한 번역이나, 사람이라면
절대 저지르지 않을 것 같은 실수도 한다.

#### 3-2. 트랜스포머 (Transformer)

RNN의 가장 큰 문제는 시계방향으로의 계산에서 병렬계산을 사용할 수 없다는 점이다. GPU는 병렬계산을 빠르게 처리하는데
용이한데, RNN 같은 경우 병렬계산이 안되어, GPU 병목이 발생한다. 이때문에, 시계열 분석에 있어, 아예 RNN을 사용하지 않는
시도가 이루어지고 있는데, 그중 가장 유명한 기법이 바로 Transformer이다. Transformer는 2017년 'Attention is all you need'라는
논문에서 처음 제시가 되었다.

> Transformer 이외에도 RNN을 대체하기 위한 연구는 활발히 진행되고 있다. 예를 들어, RNN 대신 합성곱 계층을 활용하여
> seq2seq 를 구성하는 등, GPU의 병렬계산 능력을 극대화 시킬 수 있는 신경망 모델이 많이 연구되고 있다.

Transformer는 어텐션만으로 구성된 모델로, 이중 self-attention 이라는 기술이 핵심이다.
Transformer를 이용하면 RNN 보다 계산량을 줄일 수 있고, GPU를 이용한 병렬계산의 해택을 많이 누릴 수 있다.
