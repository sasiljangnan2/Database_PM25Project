import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
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
# =====================================================================
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"[{name}]  MAE {mae:.3f} | RMSE {rmse:.3f} | R2 {r2:.3f}")
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


results = []
predictions = {}

# =====================================================================
# 모델 1) 다중선형회귀 (Baseline)
# =====================================================================
lr = LinearRegression().fit(X_train_sc, y_train)
predictions['다중선형회귀'] = lr.predict(X_test_sc)
results.append(evaluate('다중선형회귀', y_test, predictions['다중선형회귀']))

# =====================================================================
# 모델 2) Ridge 회귀 
# =====================================================================
alpha_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge_gs = GridSearchCV(Ridge(), alpha_grid, cv=5, scoring='r2').fit(X_train_sc, y_train)
ridge = ridge_gs.best_estimator_
predictions['Ridge'] = ridge.predict(X_test_sc)
print(f"  (Ridge 최적 alpha = {ridge_gs.best_params_['alpha']})")
results.append(evaluate('Ridge', y_test, predictions['Ridge']))

# =====================================================================
# 모델 3) Lasso 회귀
# =====================================================================
lasso_gs = GridSearchCV(Lasso(max_iter=10000), alpha_grid, cv=5, scoring='r2').fit(X_train_sc, y_train)
lasso = lasso_gs.best_estimator_
predictions['Lasso'] = lasso.predict(X_test_sc)
print(f"  (Lasso 최적 alpha = {lasso_gs.best_params_['alpha']})")
results.append(evaluate('Lasso', y_test, predictions['Lasso']))

# =====================================================================
# 모델 4) Random Forest
# =====================================================================
rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                           random_state=42, n_jobs=-1).fit(X_train_sc, y_train)
predictions['RandomForest'] = rf.predict(X_test_sc)
results.append(evaluate('RandomForest', y_test, predictions['RandomForest']))


# =====================================================================
# 성능 비교표
# =====================================================================
result_df = pd.DataFrame(results)
print("\n=== 모델 성능 비교 ===")
print(result_df.round(3).to_string(index=False))


# =====================================================================
# 시각화: (1) R2 비교  (2) RF 예측 vs 실제  (3) RF 피처 중요도
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (1) R2 막대 비교
axes[0].bar(result_df['Model'], result_df['R2'],
            color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
axes[0].set_title('모델별 R2 비교')
axes[0].set_ylabel('R2'); axes[0].set_ylim(0, 1)
for i, v in enumerate(result_df['R2']):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center')
axes[0].tick_params(axis='x', rotation=20)

# (2) Random Forest 예측 vs 실제
axes[1].scatter(y_test, predictions['RandomForest'], alpha=0.3, s=10, color='#8172B2')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_xlabel('실제 PM2.5'); axes[1].set_ylabel('예측 PM2.5')
axes[1].set_title('Random Forest: 예측 vs 실제')

# (3) Random Forest 피처 중요도
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
axes[2].barh(imp.index, imp.values, color='#55A868')
axes[2].set_title('Random Forest 피처 중요도')
axes[2].set_xlabel('중요도')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=200)
print("\nmodel_comparison.png 저장됨")
