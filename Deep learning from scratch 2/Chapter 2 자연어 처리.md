## 자연어 처리 방식
### 1. 시소러스
시소러스는 (기본적으로는) 유의어 사전으로, '뜻이 같은 단어(동의어)'나 '뜻이 비슷한 단어(유의어)'가 한 그룹으로
분류되어 있다. 보통 각 단어의 관계를 그래프 구조로 정의한다.

#### 시소러스의 문제점은 다음과 같다.
* 시대 변화에 대응하기 어렵다.
* 사람을 쓰는 비용이 크다.
* 단어의 미묘한 차이를 표현할 수 없다.

### 2. 통계 기반 기법
통계 기반 기법은 대량의 텍스트 데이터(corpus)를 사용한다. 이러한 다량의 텍스트에는
자연어에 대한 사람의 지식이 충분히 담겨있으므로, 이러한 말뭉치에서 자동으로 그 핵심을 추출하는 것이다.

#### 파이썬으로 말뭉치 전처리하기
```python
def preprocess(text):
  text = text.lower()
  text = text.replace('.', ' .')
  words = text.split(' ')
  
  word_to_id = {}
  id_to_word = {}
  for word in words:
    if word not in word_to_id:
      new_id = len(word_to_id)
      word_to_id[word] = new_id
      id_to_word[new_id] = word
  
  corpus = np.array([word_to_id[w] for w in words])
  
  return corpus, word_to_id, id_to_word
```

#### 단어의 분산 표현
단어를 하나의 벡터로 표현하는 것을 '단어의 분산 표현'(distributional representation)이라고 한다.

#### 단어의 분포 가설
'단어의 의미는 주변 단어에 의해 형성된다'라는 가설이 바로 단어의 분포 가설(distributional hypothesis)이라고 한다.
분포 가설이 말하고자 하는 바는 매우 간단하다. 단어 자체에는 의미가 없고, 그 단어가 사용된 '맥락'이 의미를 형성한다는 것이다.

#### 동시발생 행렬
단어의 분포 가설을 활용하여 각 단어의 동시발생 행렬을 만들어 볼것이다.

```python
def create_co_matrix(corpus, vocab_size, window_size=1):
  corpus_size = len(corpus)
  co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
  
  for idx, word_id in enumerate(corpus):
    for i in range(1, window_size + 1):
      left_idx = idx - i
      right_idx = idx + i
      
      if left_idx >= 0:
        left_owrd_id = corpus[left_idx]
        co_matrix[word_id, left_word_id] += 1
      if right_idx < corpus_size:
        right_word_id = corpus[right_idx]
        co_matrix[word_id, right_word_id] += 1
  return co_matrix
```

#### 벡터 간 유사도
벡터 사이의 유사도를 측정하는 방법은 다양하다. 대표적으로 벡터의 내적이나, 유클리드 거리 등을 꼽을 수 있다.
그 외에도 다양하지만, 단어 벡터의 유사도를 나타낼 때는 코사인 유사도를 자주 이용한다.

```python
def cos_similarity(x, y, eps=1e-8):
  nx = x / np.sqrt(np.sum(x**2) + epx)
  ny = y / np.sqrt(np.sum(y**2) + eps)
  return np.dot(nx, ny)
```
이 함수를 활용하면 단어 벡터의 유사도를 다음과 같이 구할 수 있다.

```python
text = 'You say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = creat_to_matrix(corpus, vocab_size)

c0 = C[word_to_id['You']]
c1 = C[word_to_id['i']]
print(cos_similarity(c0, c1))
```


