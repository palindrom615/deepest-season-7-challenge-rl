## Tasks

### 버그 수정

1. learner.py의 49번째 라인에서

```python
        self.schema = DataSchema(
            names=["prev_state", "action", "reward", "state", "done"],
            shapes=[(self.input_size,), (1,), (1,), (self.input_size,), (1,)],
            dtypes=[np.int64, np.int64, np.float32, np.float32, np.float32],
        )
```

라고 되어 있는데, gym의 cartpole 문제의 경우 state는 (4, 1)의 float 값, done은 boolean 값으로, 위 dtypes를 `[np.float32, np.int64, np.float32, np.float32, np.bool]`로 고쳐야 한다.

diff: https://github.com/palindrom615/deepest-season-7-challenge-rl/commit/9bec9fe7c96b891b708da666cad6196829c4c12c

2. learner.py의 74번째 라인에서

```python
        target = reward + done * self.gamma * selected_tq
```

라고 되어 있다. `done`은 (batch_size, 1) dimension이고 `selected_tq`는 (batch_size) dimension으로 둘을 그냥 * 해버리면 target의 결과값으로 (batch_size, batch_size) dimension이 들어가게 된다. 이는 우리가 원하는 바가 아니므로 element-wise 곱셈을 제대로 하도록 `selected_tq`를 `done`과 같은 차원으로 reshape해줄 필요가 있다.

diff: https://github.com/palindrom615/deepest-season-7-challenge-rl/commit/81a27b43a06a49a0fefb0a2964ba4a06e1f136b3

3. 마찬가지로 learner.py의 71, 72번째 라인에서 (batch_size, 2) dimension인 네트워크의 아웃풋 값을 결과로 만드는 과정에서 둘 다 dim=-1을 dim=1로 고쳐야 한다. 또한 torch.max 함수에 dimension을 명시해주면 앞에 결과 값, 뒤에 indices 값이 들어가므로 selected_tq를 뒤의 값이 아니라 앞의 값을 받아 오도록 고쳐야 한다.

diff: https://github.com/palindrom615/deepest-season-7-challenge-rl/commit/ec276e7bc2023918add3dee3d693131cc5ea9de3

4. learner.py의 117~119 라인에서 env.step으로 받아온 done이 true이면 새로운 에피소드를 시작하도록 break를 하는데, 그게 replay 메모리에 넣는 것보다 위에 있어서 done이 True인 경우가 메모리에 못들어가고 있다.

DQN에서는 마지막이 있어야 reward의 discounted 값이 제대로 계산되므로, 이 분기문을 아래로 내려줘야 한다.

diff: https://github.com/palindrom615/deepest-season-7-challenge-rl/commit/2812053cb78d867d867be93c09091858891ee23e

5. learner.py의 74 라인에서 done = True일 때 target = reward이고, False일 때(즉 다음 state가 있을 때) 다음 state의 q값을 discount해서 reward에 더해야 한다. 그런데 원래 코드처럼 done 텐서를 그냥 곱해버리면 done이 True일 때는 1, False일 때는 0이 곱해져 반대로 학습을 하게 된다. 이를 바로 잡기 위해 done을 negate해 줘야 한다.

diff: https://github.com/palindrom615/deepest-season-7-challenge-rl/commit/f0a7c04f72a549d0a330fb95aaadfdc31143d032

### L1 loss 사용하기

### Load & Save & Play 구현

구현 완료.

https://github.com/palindrom615/deepest-season-7-challenge-rl/commit/19b75598ca2c760bb3864b38238422f499076cfe