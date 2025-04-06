#!/bin/bash
# 가상 거래 테스트를 위한 명령 실행 예제

# 필요한 디렉토리 생성
mkdir -p data/sandbox
mkdir -p data/evaluation

# 1. 가상 계좌 생성
echo "=== 가상 거래 계좌 생성 ==="
python src/crew_chain/cli/evaluate.py account create --balance 50000 --save data/sandbox/account_id.txt
ACCOUNT_ID=$(cat data/sandbox/account_id.txt)
echo "계좌 ID: $ACCOUNT_ID"
echo ""

# 2. 단일 전략 평가
echo "=== 단일 전략 평가(3일 시뮬레이션) ==="
python src/crew_chain/cli/evaluate.py evaluate \
  --account-id $ACCOUNT_ID \
  --symbol bitcoin \
  --days 3 \
  --output data/evaluation/single_strategy_results.json
echo ""

# 3. 여러 전략 비교
echo "=== 여러 전략 비교 ==="
python src/crew_chain/cli/evaluate.py compare \
  --account-id $ACCOUNT_ID \
  --symbol ethereum \
  --days 2 \
  --strategies-file examples/strategy_comparison.json \
  --output data/evaluation/strategy_comparison_results.json
echo ""

# 4. 백테스팅
echo "=== 히스토리컬 데이터로 백테스팅 ==="
python src/crew_chain/cli/evaluate.py backtest \
  --account-id $ACCOUNT_ID \
  --symbol bitcoin \
  --data-file data/historical/bitcoin_sample.csv \
  --output data/evaluation/backtest_results.json
echo ""

# 5. 수동 거래 실행
echo "=== 수동 거래 실행 ==="
python src/crew_chain/cli/evaluate.py account trade \
  --account-id $ACCOUNT_ID \
  --symbol bitcoin \
  --action buy \
  --amount 5000 \
  --position-type long
echo ""

# 6. 계좌 요약 확인
echo "=== 계좌 상태 확인 ==="
python src/crew_chain/cli/evaluate.py account summary \
  --account-id $ACCOUNT_ID
echo ""

echo "=== 모든 테스트 완료 ==="
echo "결과는 data/evaluation 디렉토리에 저장되었습니다." 