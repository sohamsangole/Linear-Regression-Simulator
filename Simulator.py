import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
import myLinearRegression as mLinReg

df = pd.read_csv("dataset/Salary_Data.csv")

X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=69, test_size=0.2, shuffle=True)

w_final, b_final, J, p = mLinReg.gradient_descent(X_train, y_train, 0, 0, 0.01, 30)

print(p)
w_history = [elem[0] for elem in p]
b_history = [elem[1] for elem in p]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# Plot the first graph
ax[0].scatter(w_history, J)
ax[0].set_xlabel('w')
ax[0].set_ylabel('Cost')
ax[0].set_title('Cost vs w')

line, = ax[1].plot([], [], label='Regression Line')

# Plot the second graph
ax[1].plot(X_train, w_final * X_train + b_final, label='Final Line', color='red')
ax[1].set_xlabel('Years of Experience')
ax[1].set_ylabel('Salary')
ax[1].set_title('Gradient Descent')
ax[1].legend()

# Adjust the spacing
plt.tight_layout()


def update(frame):
    w, b = p[frame]
    y_pred = w * X_train + b
    line.set_data(X_train, y_pred)
    ax[1].set_title('Gradient Descent')

    w_history_frame = w_history[:frame + 1]
    J_frame = J[:frame + 1]
    ax[0].cla()  # Clear the previous plot
    ax[0].scatter(w_history_frame, J_frame)
    ax[0].set_ylim([min(J), max(J)])
    ax[0].set_xlim([min(w_history), max(w_history)])
    ax[0].set_xlabel('w')
    ax[0].set_ylabel('Cost')
    ax[0].set_title('Cost vs w')
    return line,


animation = FuncAnimation(fig, update, frames=len(p), interval=0.001, repeat_delay=1000)
animation.save('gradient_descent_animation.gif', writer='pillow')
plt.show()
