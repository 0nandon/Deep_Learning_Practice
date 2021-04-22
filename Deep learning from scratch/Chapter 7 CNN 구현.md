## numpy로 CNN 구현
만약 합성곱 연산을 곧이곧대로 구현할려면 for문을 겹겹히 쌓아야 한다. 그러나, 넘파이에 for문을 사용하면 성능이 떨어진다.
그래서 보통 합성곱 연산을 구현할 때는, im2col 함수를 사용한다. im2col은 입력 데이터를 필터링 하기 좋게 전개하는 함수이다.

```python
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
```
이 함수를 이용해서 CNN을 구현한다.

```python
class CNN:
    def __init__(self, x, w, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        
        out_h = int((H + 2*self.pad - FH) / self.stride)
        out_w = int((W + 2*self.pad - FW) / self.stride)
        
        self.W = self.W.reshape(FN, -1).T
        x = im2col(x, FH, FW)
        out = np.dot(x, self.W) + self.b
        
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out
```
