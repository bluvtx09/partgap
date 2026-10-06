# 수요 확인: 커뮤니티 글 (사전 등록)

작성 2026-10-02. 판정 기준을 글을 올리기 전에 고정한다.

## 무엇을 확인하나

"내 팔의 서보를 5분 재서 시뮬 값을 받는 것"을 쓸 사람이 있는지 본다.
글만 올리면 반응이 적을 수 있다(H3: 기존 불만 글의 댓글 0~2개). 그래서 **바로 써 볼 수 있는 무료 도구**를 같이 준다.
`partgap-armcheck`는 자기 LeRobot 데이터셋 이름을 넣으면, 그 팔의 멈춤 오차가 공개된 팔 193대 중 몇 등인지 알려 준다.
도구를 실제로 돌려 보고 결과를 남기는 사람 수가 첫 번째 신호다.

## 올리는 순서

1. **partgap 저장소에 최신 커밋을 먼저 올린다.** 글의 링크와 설치 명령이 저장소를 가리킨다. 아직 403 오류로 못 올린 상태라, 권한을 고치거나 번들을 직접 올려야 한다.
2. 글 1을 Reddit r/robotics에 올린다.
3. 같은 날 글 2를 Hugging Face 포럼(discuss.huggingface.co)과 LeRobot Discord에 올린다.
4. 올린 날을 아래 기록표에 적는다. 14일 뒤에 센다.

댓글에는 직접 답한다. 질문에 답하는 것도 신호를 키운다.

## 판정 기준 (첫 글 후 14일)

세는 것은 **서로 다른 사람 수**다. 같은 사람이 여러 번 쓰면 1명이다.

| 신호 | 세는 법 |
|---|---|
| A. 도구 사용 | 자기 팔의 결과 숫자를 남긴 사람 |
| B. 쓰겠다 | 글의 질문 2에 "yes"라고 답한 사람 |
| C. 돈 | 질문 2에서 가격을 적었고 그 값이 10달러 이상인 사람 |

| 결과 | 판정 | 다음 |
|---|---|---|
| B 10명 이상 그리고 C 3명 이상 | 수요 있음 | 서보 측정 실험 결과에 맞춰 키트 시제품으로 간다 |
| B 3~9명, 또는 B 10명 이상인데 C 3명 미만 | 약함 | 무료 도구와 측정 방법만 공개하고, 연구 결과(논문·세특)로 쓴다 |
| B 3명 미만 | 없음 | 보정 키트 사업안은 접는다 |

A는 판정에 넣지 않고 보고만 한다. 다만 A가 0이면 도구가 안 돌아가는지 먼저 확인한다.

## 기록표

| 날짜 | 어디 | 링크 | A | B | C | 메모 |
|---|---|---|---|---|---|---|
| | r/robotics | | | | | 2026-10-06 시도 실패: 계정 나이 제한(계정 생성 1일). 프로필에만 올라감(https://www.reddit.com/user/bluvtx09/comments/1wyw5n4/), 집계 제외 |
| | HF 포럼 | | | | | |
| | LeRobot Discord | | | | | |

---

## 글 1: Reddit r/robotics

**Title:** I compared 193 public SO-100 arms: with the same servo model, how precisely the arm stops varies a lot. Check yours with one command

I've been working on why robot sims don't match real arms. One question kept coming up: if everyone uses the same servo, can one sim model fit every arm?

So I took public LeRobot datasets from Hugging Face, one per uploader, and looked at the base joint (shoulder_pan) of 193 SO-100 arms.
For every pause in every episode, I measured how far the servo stops short of its target. Then I compared stops after moving one way with stops after moving the other way. The difference is the hysteresis.

What I found:

- The median arm stops with 0.41° of hysteresis. The middle half of the arms range from 0.19° to 0.71°, and the top 10% are at about 1.2° or more.
- Within each dataset it's consistent, not random noise. When I split each dataset's episodes into two halves, the two halves ranked the arms almost the same (Spearman 0.86). That shows the value is stable per dataset; it doesn't by itself prove the servo is the cause.
- It isn't explained by how fast people moved the arm (correlation 0.07). Newer datasets tend to show less of it (correlation with recording date −0.32), which could mean settings or kit versions changed over time.

So a sim tuned on one arm will likely be off on another. I can't yet say whether this comes from the servos themselves or from settings, voltage version and assembly. I'm testing that next with several servos on one bench.

**Check your own arm** (the reference is SO-100 datasets recorded in degrees; SO-101 uses the same servo and the tool runs on it, but that comparison isn't validated yet):

```
pip install "partgap[armcheck] @ git+https://github.com/bluvtx09/partgap"
partgap-armcheck <your_hf_user>/<your_dataset>
```

It downloads up to 24 episodes and tells you where your arm falls among the 193.

Two questions:
1. If you run it, could you post your number and your setup (SO-100 or SO-101, 5 V/7.4 V/12 V servos)? I'd like to see whether voltage or kit version explains the spread.
2. If a 5-minute test gave you sim parameters (MuJoCo friction and damping) fitted to *your* arm's servos, would you use it? Yes / maybe / no. And if yes, what would you pay for it, if anything?

Method, data and code: https://github.com/bluvtx09/partgap/tree/main/studies/business_validation

---

## 글 2: Hugging Face 포럼 / LeRobot Discord

**Title:** How different is your SO-100/101 arm? A one-command check against 193 public SO-100 arms

Hi all. I measured the stopping hysteresis of shoulder_pan on 193 public SO-100 datasets (one per uploader).
It varies a lot between arms (median 0.41°, top 10% at about 1.2° or more) and it's consistent within each dataset. I don't know yet how much is the servo itself versus settings and assembly.
That matters if you train in sim or share policies between arms.

You can check your own arm from a dataset you've recorded:

```
pip install "partgap[armcheck] @ git+https://github.com/bluvtx09/partgap"
partgap-armcheck <your_hf_user>/<your_dataset>
```

It needs angles in degrees (`use_degrees=True`); the reference is SO-100 only, so SO-101 results are a rough comparison. If you try it, I'd love to hear your number and your servo voltage.

One more question: would a 5-minute servo test that gives you sim parameters for *your* arm be useful to you? Yes / maybe / no, and what would it be worth to you?

Details: https://github.com/bluvtx09/partgap/tree/main/studies/business_validation
