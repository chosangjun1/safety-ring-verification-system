import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# pip 명령 실행
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])


# 1. 데이터 생성
np.random.seed(42)
data_size = 1000

# 전기 안전 데이터를 시뮬레이션 (가상 데이터 생성)
data = pd.DataFrame({
    'Voltage': np.random.normal(220, 10, data_size),  # 전압 (평균 220V, 표준편차 10V)
    'Current': np.random.normal(10, 2, data_size),    # 전류 (평균 10A, 표준편차 2A)
    'Leakage_Current': np.random.uniform(0, 1, data_size),  # 누설 전류 (0~1A)
    'Temperature': np.random.normal(35, 5, data_size),  # 온도 (평균 35°C, 표준편차 5°C)
    'Power_Consumption': np.random.normal(2000, 500, data_size),  # 전력 소비량 (평균 2000W, 표준편차 500W)
    'Equipment_Usage_Time': np.random.uniform(0, 24, data_size),  # 장비 사용 시간 (0~24시간)
    'Accident_History': np.random.randint(0, 5, data_size)  # 과거 사고 기록 횟수 (0~4회)
})

# 타겟 변수: 사고 발생 여부 (0: 정상, 1: 사고)
data['Accident_Occurred'] = (data['Voltage'] > 240) | \
                            (data['Current'] > 15) | \
                            (data['Leakage_Current'] > 0.8) | \
                            (data['Temperature'] > 50) | \
                            (data['Accident_History'] > 2)
data['Accident_Occurred'] = data['Accident_Occurred'].astype(int)

# 2. 데이터 분리
X = data.drop('Accident_Occurred', axis=1)
y = data['Accident_Occurred']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 학습
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 4. 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy)

# 5. 시각화 (특성 중요도)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Feature Importance for Accident Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()