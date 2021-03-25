## Callback
딥러닝을 시행함에 있어, 최적의 검증손실을 유도하는 에포크를 찾아내는 것은 매우 어려운 일이다. 보통은 검증손실이 더 이상 향상되지 않을 때,
훈련을 멈추는데, 이때 callback을 사용하면 매우 편리하다. callback은 모델의 상태와 성능에 대한 모든 정보에 접근하고 훈련 중지, 모델 저장, 가중치 적재 또는
모델 상태 변경 등을 처리할 수 있다.

다음은 Callback을 사용하는 몇가지 사례이다.
* 모델 체크포인트 저장 : 훈련하는 동안 여러 지점에서 모델의 현재 가중치를 저장한다.
* 조기 종료(early stopping) : 검증 손실이 더 이상 향상되지 않을 때 훈련을 중지한다.
* 훈련하는 동안 하이퍼 파라미터 값을 동적으로 조정한다. (ex : 옵티마이저 learning rate, l2 규제 정도, etc)
* 훈련과 검증 지표를 로그에 기록하거나 모델이 학습한 표현이 업데이트될 때마다 시각화한다.

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
