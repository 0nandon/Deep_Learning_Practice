## Callback
딥러닝을 시행함에 있어, 최적의 검증손실을 유도하는 에포크를 찾아내는 것은 매우 어려운 일이다. 보통은 검증손실이 더 이상 향상되지 않을 때,
훈련을 멈추는데, 이때 callback을 사용하면 매우 편리하다. callback은 모델의 상태와 성능에 대한 모든 정보에 접근하고 훈련 중지, 모델 저장, 가중치 적재 또는
모델 상태 변경 등을 처리할 수 있다.

다음은 Callback을 사용하는 몇가지 사례이다.
* 모델 체크포인트 저장 : 훈련하는 동안 여러 지점에서 모델의 현재 가중치를 저장한다.
* 조기 종료(early stopping) : 검증 손실이 더 이상 향상되지 않을 때 훈련을 중지한다.
* 훈련하는 동안 하이퍼 파라미터 값을 동적으로 조정한다. (ex : 옵티마이저 learning rate, l2 규제 정도, etc)
* 훈련과 검증 지표를 로그에 기록하거나 모델이 학습한 표현이 업데이트될 때마다 시각화한다.


## ModelCheckPoint와 EarlyStopping 콜백

```python
import keras

callbacks_list = [
  keras.callbacks.EarlyStopping( # 성능 향상이 멈추면 훈련을 중지한다.
    monitor = 'val_acc', # 모델의 검증 정확도를 모니터링한다.
    patience = 1, # 1 에포크보다 더 길게 검증 정확도가 향상되지 않으면, 훈련을 중지한다.
  ),
  keras.callbacks.ModelCheckpoint(
    filepath = 'my_model.h5',
    monitor = 'val_loss',
    save_best_only = True, # 훈련하는 동안 가장 최적의 모델만 저장한다.
  )
]

#...

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['acc'])
model.fit(x, y,
          epochs = 10,
          callbacks = callbacks_list,
          validation_data = (x_val, y_val))
```
## ReduceLROnPlateau 콜백
이 콜백을 사용하면 검증 손실이 향상되지 않을 때, 학습률을 작게 할 수 있다. 손실 곡선이 평탄할 때,
학습률을 작게 하거나 크게 하면 훈련 도중 지역 최솟값을 효과적으로 빠져나올 수 있다.

```python
callbacks_list = [
  keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_acc',
    factor = 0.1, # 콜백이 호출될 때, 학습률을 10배 줄인다.
    patience = 10, # 검증 손실이 10번동안 향상되지 않으면 콜백이 호출됨
  )
]
```

## 자신만의 콜백 만들기
내장 콜백에서 제공하지 않은 특수한 행동이 훈련 도중 필요하다면 자신만의 콜백을 만들 수 있다.

* on_epoch_begin : 각 에포크가 시작할 때 호출한다.
* on_epoch_end : 각 에포크가 끝날 때 호출한다.
* on_batch_begin : 각 배치 처리가 시작되기 전에 호출한다.
* on_batch_end : 각 배치 처리가 끝난 후에 호출한다.
* on_train_begin : 훈련이 시작될 때 호출한다.
* on_train_end : 훈련이 끝날 때 호출한다.

```python
import keras
import numpy as np

class ActivationLogger(keras.callbacks.Callback):
  def set_model(self, model): # 호출하는 모델에 대한 정보를 전달하기 위해 훈련 전에 호출한다.
    self.model = model
    layer_outputs = [layer.outputs for layer in model.layers]
    self.activations_model = keras.models.Model(inputs = model.input, outputs = layer_outputs)
  
  def on_epoch_end(self, epoch, logs = None): # 매 에포크가 끝날 때 마다 호출한다.
    if self.validation_data is None:
      raise RuntimeError('Requires validation_data.')
      
    validation_sample = self.validation_data[0][0:1]
    activations = self.activations_model.predict(validation_sample)
    
    f = open('activations_at_epoch_' + str(epoch) + '.npz', 'wb')
    np.savez(f, activations)
    f.close()
```
