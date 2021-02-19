
## 다중 입력 모델
keras는 함수형 API(functional api)를 통해 다중 입력 모델을 구현할 수 있다. 아래와 같은 함수들이 사용된다.
* keras.layers.add
* keras.layers.concatenate

아래는 예시코드이다.
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, concatenate, Dense, Activation

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape = (None,), dtype='int32', name='text')

embedded_text = Embedding(question_vocabulary_size, 64)(text_input)
encoded_text = LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32', name='text')
encoded_question = LSTM(32)(question_input)

concatenated = concatenate([encoded_text, encoded_question])
answer = Dense(1)(answer_vocabulary_size)(concatenate)
answer = Activation('softmax')(answer)
```
