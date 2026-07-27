# 추천 품질 개선 작업 완료 보고 — 2026-07-25

## 결론

이번 historical 연구에서 활성 추천에 넣을 만큼 검증된 새 모델은 **0개**다.
후보 5축을 끝까지 비교하고 독립 재계산했지만, 모두 “상승률을 높이면 하락률도 함께
높아지는” 문제를 해결하지 못했다. 따라서 성능이 나쁜 후보를 억지로 Telegram에 섞지
않았고, 활성 R1·알림 문구·자동 스케줄·원장은 그대로 유지했다.

대신 앞으로 추천을 제대로 고를 수 있도록 측정 경로를 완성했다. 다음 정상 실행부터
사용자가 실제로 받을 수 있었던 시각을 기준으로 전 유니버스 약 100개 점수와 이후
24시간 결과가 매일 쌓인다.

## 사용자가 실제로 받는 것

1. 기존 R1 preopen/open과 pump v2 알림은 기존 형식으로 계속 온다.
2. 자동 주문은 없고 사용자가 직접 거래한다.
3. 같은 슬롯의 Telegram과 원장은 한 번 만든 동일 snapshot을 사용한다.
4. 알림 성공 시각이 기록되며, 예를 들어 09:10 발송이면 09:15부터 다음 날
   09:15까지 정확히 96개 15분봉으로 평가한다.
5. 매일 top3뿐 아니라 그날 비교 가능했던 약 100개 전 종목의 상방·하방 점수와
   D-1 피처가 저장된다.
6. 하루가 완전히 지난 뒤 다음을 자동 계산한다.
   - `up5/up10/up20`
   - `dn3/dn5/dn10`
   - `up10 AND NOT dn5` 안전상승
   - TP5/SL3 중 무엇이 먼저 왔는지
   - MFE/MAE와 비용 0.15% 차감 수익
   - R1 top-N 대 전 유니버스·유동성 매칭·변동성 band baseline
7. Telegram 실패, 핵심 수집 실패, 경로 불완전은 성공으로 숨기지 않고
   receipt·exit code·실패 알림으로 드러난다.

아직 실제 새 forward snapshot/label은 0개다. 코드는 오늘 완성됐으므로 다음 정상
스케줄부터 생성된다. 과거 데이터를 오늘 다시 계산한 값은 `scheduled_replay`로
분리되어 실제 forward 성과에 섞이지 않는다.

## 후보별 최종 판정

| 후보 | 확인된 장점 | 실패한 이유 | 판정 |
|---|---|---|---|
| 하방 top-third veto | `dn5 -8.61%p` | safe-up `-2.25%p`, net 개선 `+0.009%p`로 사실상 0 | REJECT |
| Binance 상방 head | 전체 AUC 약 `0.745` | top3 `dn5 68.91%`, net `-0.410%/픽`, 당일 fresh Binance 값은 R1 발송 뒤 도착 | REJECT |
| direct safe-up head | R1 대비 safe-up `+5.65%p` | `dn5 +15.07%p`, net 개선 불확실 | REJECT |
| fixed first-passage | R1 대비 safe-FP `+5.89%p` | `dn5 +20.12%p`, SL-first `+12.20%p`, net 개선 불확실 | REJECT |
| ATR first-passage | 고정 장벽보다 변동성 복사 감소 | discovery `dn5 +11.52%p`, net 악화 | REJECT |
| downside semivol 4개 | core 대비 `dn5 -2.17%p` | CI가 0 포함, safe-FP 변화 0, net·AUC 악화 | REJECT |

단순 lowest-ATR는 `dn5 4.88%`로 하방만 보면 좋지만 `up10 0.61%`,
safe-FP `0.41%`, net `-0.360%/픽`이었다. 이는 사용자가 원하는 “안 빠지면서
오를 후보”가 아니라 “거의 움직이지 않는 후보”다.

## 왜 새 모델을 연결하지 않았는가

상방 head 자체는 완전히 죽지 않았다. repaired up10/first-passage 계열의
full-universe AUC는 약 `0.70~0.74`였다. 문제는 이 점수의 상단이 고변동 종목에
몰려 상승 가능성과 큰 하락 가능성을 함께 높인다는 것이다.

하방을 강하게 자르면 반대 문제가 생겼다. 하락은 줄지만 상승도 거의 사라졌다.
semivolatility까지 추가해도 두 축을 유의하게 분리하지 못했다.

따라서 이 시점에 임계값·비율·가중치를 더 만들어 같은 마지막 180일에 맞추는 것은
개선이 아니라 사후 데이터 마이닝이다. 통과 후보가 없으므로 SHADOW도 추가하지
않는 것이 추천 품질을 지키는 결정이다.

## 이제 남은 일

역사 데이터 식 탐색은 여기서 닫는다. 다음 판단은 새 forward 표본에서만 한다.

- 실제 전달된 추천과 전 유니버스 score가 매일 정상 생성되는지 확인
- complete/partial/flat-filled 비율과 누락 편향 감시
- R1 top3가 같은 날 유니버스·유동성 매칭 baseline보다
  safe-up을 높이면서 `dn5`와 비용 차감 net을 악화하지 않는지 확인
- 충분한 날짜와 양·음성 표본이 쌓인 뒤에만 새 head 또는 veto를
  record-only SHADOW로 다시 평가
- 활성 모델·라벨·알림 형식 변경은 별도 사용자 승인 후 진행

운영 코드에서 남은 수동 작업은 새 systemd unit을 `/etc/systemd/system`에
반영하는 sudo 1회다. 소스와 테스트는 준비됐지만 이 설치는 시스템 스케줄을
바꾸므로 자동 수행하지 않았다.

## 검증과 근거

- 전체 테스트: `177 passed`
- 표적 경로·snapshot·label·evaluator 테스트: 통과
- Python 컴파일과 변경 범위 lint: 통과
- first-passage 핵심 산출물 9개: 새 cache 실행과 cache-read 실행의
  byte SHA-256 일치
- semivol 산출물: 연속 2회 byte SHA-256 일치
- 독립 재계산은 연구 summary가 아니라 row artifact와 SQLite 경로에서 다시 수행

상세 근거:

- `_workspace/challenger_quant_evaluator_verdict_v1.md`
- `_workspace/first_passage_head_challenger_v1.md`
- `_workspace/semivol_joint_challenger_v1.md`
- `_workspace/safeup_head_challenger_v1.md`
- `_workspace/upside_head_challenger_v1.md`
- `_workspace/downside_veto_challenger_v1.md`
- `ADDITIONAL_IDEAS.md`
