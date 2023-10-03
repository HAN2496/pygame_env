from scipy.interpolate import interp1d
import numpy as np

# 예시 데이터
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])  # y = x^2

# 선형 보간 객체 생성
f = interp1d(x, y)

# 보간 값을 얻기 위해 함수처럼 호출
print(f(1.5))  # 1.5 지점에서의 보간 값
