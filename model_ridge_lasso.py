"""
참고사항 : GridSearchCV로 alpha를 탐색한 결과 Lasso는 0.001, Ridge는 1에서 최적이었으며, 규제 적용 후에도 성능이 기본 선형회귀와 동일.
-> 피처 수가 적어 모델이 단순하고 과적합이 발생하지 않았음을 의미.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import platform

# 시각화 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


# csv 읽기 -> 피처/타겟 분리 -> 분할 -> 정규화
df = pd.read_csv('airkorea_2024_seongbuk_cleaned.csv', encoding='utf-8-sig')

target = '초미세먼지(PM25)'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# =====================================================================
# 평가 함수
# MAE  : 평균 절대 오차 (작을수록 좋음)
# RMSE : 큰 오차에 민감 (작을수록 좋음)
# R2   : 결정계수 (1에 가까울수록 좋음)
# =====================================================================
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"MAE  : {mae:.4f}   RMSE : {rmse:.4f}   R2 : {r2:.4f}")
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


results = []

# =====================================================================
# 모델 1) 다중선형회귀 (Baseline) - 규제 없음, 비교 기준선
# =====================================================================
lr = LinearRegression().fit(X_train_sc, y_train)
results.append(evaluate('다중선형회귀 (Baseline)', y_test, lr.predict(X_test_sc)))


# =====================================================================
# 모델 2) Ridge 회귀 (L2 규제) + alpha 튜닝
# ---------------------------------------------------------------------
# alpha: 규제 강도. GridSearchCV로 탐색하여 최적 값 선택
#   - cv=5: 학습 데이터를 5조각으로 나눠 교차검증 (검증셋 대체)
#   - scoring='r2': R2가 가장 높은 alpha를 최적으로 판단
# =====================================================================
alpha_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

ridge_gs = GridSearchCV(Ridge(), alpha_grid, cv=5, scoring='r2')
ridge_gs.fit(X_train_sc, y_train)
print(f"\n[Ridge 튜닝] 최적 alpha = {ridge_gs.best_params_['alpha']}")
ridge_best = ridge_gs.best_estimator_
results.append(evaluate('Ridge 회귀 (L2, 튜닝)', y_test, ridge_best.predict(X_test_sc)))


# =====================================================================
# 모델 3) Lasso 회귀 (L1 규제) + alpha 튜닝
# ---------------------------------------------------------------------
# L1 규제는 일부 계수를 0으로 만들어 변수 선택 효과가 있음
# =====================================================================
lasso_gs = GridSearchCV(Lasso(max_iter=10000), alpha_grid, cv=5, scoring='r2')
lasso_gs.fit(X_train_sc, y_train)
print(f"\n[Lasso 튜닝] 최적 alpha = {lasso_gs.best_params_['alpha']}")
lasso_best = lasso_gs.best_estimator_
results.append(evaluate('Lasso 회귀 (L1, 튜닝)', y_test, lasso_best.predict(X_test_sc)))


# =====================================================================
# 회귀계수 비교 (정규화 덕분에 계수 크기로 영향력 비교 가능)
# =====================================================================
coef_df = pd.DataFrame({
    'Feature': X.columns,
    '다중선형회귀': lr.coef_,
    'Ridge': ridge_best.coef_,
    'Lasso': lasso_best.coef_,
})
print("\n=== 회귀계수 비교 ===")
print(coef_df.round(4).to_string(index=False))


# =====================================================================
# 성능 요약
# =====================================================================
result_df = pd.DataFrame(results)
print("\n=== 모델 성능 요약 ===")
print(result_df.round(4).to_string(index=False))


# =====================================================================
# 시각화 1) 예측 vs 실제 (Ridge 기준)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

y_pred_ridge = ridge_best.predict(X_test_sc)
axes[0].scatter(y_test, y_pred_ridge, alpha=0.3, s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('실제 PM2.5')
axes[0].set_ylabel('예측 PM2.5')
axes[0].set_title('Ridge 회귀: 예측값 vs 실제값')

# 시각화 2) 회귀계수 비교 막대그래프
coef_plot = coef_df.set_index('Feature')
coef_plot.plot(kind='barh', ax=axes[1])
axes[1].set_title('회귀계수 비교 (영향력)')
axes[1].set_xlabel('계수 크기')
axes[1].axvline(0, color='gray', linewidth=0.8)

plt.tight_layout()
plt.savefig('ridge_lasso_result.png', dpi=200)
print("\nridge_lasso_result.png 저장됨")
