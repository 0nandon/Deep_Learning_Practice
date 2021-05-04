## CBOW의 확률적 해석
CBOW 모델을 확률 표기법으로 기술해 보자. CBOW 모델이 하는 일은 맥락이 주면
타깃 단어가 출현할 확률을 출력하는 것이다.
> 단어의 맥락을 다음과 같이 표시한다.<br>
> W<sub>1</sub>, W<sub>2</sub>, W<sub>3</sub>, ... W<sub>t-1</sub>, W<sub>t</sub>, W<sub>t+1</sub>, ... W<sub>T</sub>

그럼 맥락으로 W<sub>t-1</sub>, W<sub>t+1</sub>이 주어졌을 때, 타깃이 W<sub>t</sub>가 될 확률은 수식으로 다음과 같이 쓸 수 있다.
> P(W<sub>t</sub> | W<sub>t-1</sub>, W<sub>t+1</sub>)

위 식을 이용하면 CBOW 모델의 손실함수를 다음과 같이 간단하게 표현할 수 있다.
> L = -log{ P(W<sub>t</sub> | W<sub>t-1</sub>, W<sub>t+1</sub>) }

위 와 같은 표기법을 음의 로그 가능도(negative log likelihood)라고 한다.

위 식은 하나의 데이터에 대한 손실함수 이므로, 이것을 배치 단위로 확장시키면 아래와 같은 식이 된다.
> 
