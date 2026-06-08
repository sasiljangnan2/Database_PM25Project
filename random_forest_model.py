# =====================================================================
# 라이브러리 임포트
# =====================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# =====================================================================
# 시각화 폰트 설정 (한글 깨짐 방지)
# ---------------------------------------------------------------------
# - Windows: 맑은 고딕(Malgun Gothic)
# - Mac(Darwin): 애플 고딕(AppleGothic)
# =====================================================================
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


def run_random_forest():
    # =====================================================================
    # 데이터 로드
    # =====================================================================
    print("1. 데이터 로드 중...")
    df = pd.read_csv('airkorea_2024_seongbuk_cleaned.csv', encoding='utf-8-sig')

    # =====================================================================
    # 독립 변수(X)와 종속 변수(y) 분리
    # ---------------------------------------------------------------------
    # - 타겟(Target): 초미세먼지(PM25)
    # - 피처(Features): 타겟을 제외한 나머지 모든 컬럼(가스, 날짜/시간 정보 등)
    # =====================================================================
    target = '초미세먼지(PM25)'
    X = df.drop(columns=[target])
    y = df[target]

    # =====================================================================
    # Train / Test Data Split 
    # ---------------------------------------------------------------------
    # - 전체 데이터의 80%는 학습(Train)용, 20%는 평가(Test)용으로 분할
    # - random_state=42 를 지정하여 매번 동일한 결과가 나오도록 설정
    # =====================================================================
    print("2. 데이터 분할 (Train 8 : Test 2)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # =====================================================================
    # 모델 학습 (Random Forest)
    # ---------------------------------------------------------------------
    # - n_estimators: 100 (결정 트리를 100개 생성)
    # - max_depth: 10 (각 트리의 최대 깊이를 10으로 제한하여 과적합 방지)
    # - n_jobs=-1: 사용 가능한 모든 CPU 코어를 사용하여 연산 속도 향상
    # =====================================================================
    print("3. Random Forest 모델 학습 중...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # =====================================================================
    # 예측 및 성능 평가
    # ---------------------------------------------------------------------
    # - MAE (평균 절대 오차): 실제값과 예측값의 차이 절댓값 평균
    # - RMSE (루트 평균 제곱 오차): 오차 제곱의 평균에 루트를 씌운 값
    # - R2 (결정 계수): 모델이 데이터를 얼마나 잘 설명하는지 (1에 가까울수록 좋음)
    # =====================================================================
    print("4. 모델 평가 진행 중...")
    y_pred = rf_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== Random Forest Regression 성능 평가 ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    # =====================================================================
    # 변수 중요도(Feature Importance) 분석
    # ---------------------------------------------------------------------
    # - 모델이 초미세먼지를 예측할 때 어떤 변수(가스, 시간)를 가장 중요하게 
    #   참고했는지 수치화하여 데이터프레임으로 정리
    # =====================================================================
    importances = rf_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("\n=== 변수 중요도(Feature Importance) ===")
    print(importance_df.to_string(index=False))

    # =====================================================================
    # 시각화 (변수 중요도 저장)
    # ---------------------------------------------------------------------
    # - 변수 중요도를 막대그래프(Bar Plot) 형식으로 시각화
    # - 'rf_feature_importance.png' 파일로 결과물 저장
    # =====================================================================
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', hue='Feature', legend=False)
    plt.title('Random Forest - 변수 중요도 (Feature Importance)')
    plt.xlabel('중요도')
    plt.ylabel('특성 (Feature)')
    plt.tight_layout()
    
    plt.savefig('rf_feature_importance.png', dpi=300)
    print("\n[저장 완료] 'rf_feature_importance.png' 그림 파일이 생성되었습니다.")

if __name__ == "__main__":
    run_random_forest()
