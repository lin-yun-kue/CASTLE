import numpy as np
import matplotlib.pyplot as plt

# q input: -5 ~ 5
z = np.linspace(-5, 5, 500).reshape(-1, 1)  # 100 points, each as a vector
# 假設中心在0與2, 共兩個cluster，模擬"距離"
centroids = np.array([-0.1, 0, 0.1]).reshape(1, -1)

alphas = [1]

plt.figure(figsize=(12, 6))
for alpha in alphas:
    dist_diff = z - centroids  # shape: (100, 2)
    diff = (dist_diff ** 2)  # shape: (100, 2)
    numerator = 1.0 / (1.0 + (diff / alpha))
    power = (alpha + 1.0) / 2
    numerator = numerator ** power
    q = numerator / numerator.sum(axis=1, keepdims=True)  # soft assignment

    # 計算 p (target_distribution)
    q2 = q ** 2
    p = q2 / q2.sum(axis=0, keepdims=True)
    p = p / p.sum(axis=1, keepdims=True)

    plt.plot(z.flatten(), q[:, :], label=f'q, alpha={alpha}')
    plt.plot(z.flatten(), p[:, :], '--', label=f'p, alpha={alpha}')

plt.xlabel("z value (input)")
plt.ylabel("Assignment probability (q & p for center=0)")
plt.title("q and p vs input for different alpha (center=0)")
plt.legend()
plt.grid(True)
plt.show()
