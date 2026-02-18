
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd



df = pd.read_csv("StudentPerformance.csv")

#scaling the data using min/max scaling
X_min = df["Hours Studied"].min()  # 1
X_max = df["Hours Studied"].max()  # 9
Y_min = df["Performance Index"].min()  # 40
Y_max = df["Performance Index"].max()  # 99

df["hours_norm"] = (df["Hours Studied"] - X_min) / (X_max - X_min)
df["score_norm"] = (df["Performance Index"] - Y_min) / (Y_max - Y_min)

x_col = "hours_norm"
y_col = "score_norm"

# a function to find mean square error
def mse(m, b, data, x_col, y_col):
    total_error = 0
    n = len(data)
    for i in range(n):
        x = data[x_col].iloc[i]
        y = data[y_col].iloc[i]
        y_pred = m * x + b
        total_error += (y - y_pred) ** 2
    return total_error / n

# gradient decent function for the model
def gradient_decent(m_now , b_now , data , x_col , y_col , LRate):
    m_gradient = 0 
    b_gradient = 0
    n = len(data)
    for i in range(n):
        x = data[x_col].iloc[i]
        y = data[y_col].iloc[i]
        y_pred = m_now * x + b_now
        y_error = y - y_pred
        m_gradient += -(2/n) * x * (y_error)
        b_gradient += -(2/n) * (y_error)
    m_new = m_now - LRate * m_gradient
    b_new = b_now - LRate * b_gradient
    return  m_new , b_new





losses = [] # to store all the losses
prev_loss = float("inf") # to store previous loss

# values for the gradient decent
m = 0
b = 0
Learning_rate = 0.002
epochs = 2500

# main loop
for i in range(epochs):
    m, b = gradient_decent(m, b, df, x_col, y_col, Learning_rate)
    
    loss = mse(m, b, df, x_col, y_col)
    losses.append(loss)

    if abs(prev_loss - loss) < 1e-6:
        print(f"Training converged at epoch {i}")
        break

    prev_loss = loss

    if i % 50 == 0:
        print(f"Epoch {i}, Loss: {loss}")

#de normalizing the scaled values
m_denorm = m * (Y_max - Y_min) / (X_max - X_min)
b_denorm = Y_min + b * (Y_max - Y_min) - m_denorm * X_min



# plot for loss curve
plt.figure(figsize=(8,5))
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Loss Curve for My Linear Regression")
plt.show()

y_pred_custom = m_denorm * df["Hours Studied"].values + b_denorm


X = df[["Hours Studied"]].values
y = df["Performance Index"].values
model = LinearRegression()
model.fit(X,y)
m_sklearn  = model.coef_[0]
b_sklearn = model.intercept_

y_pred_sklearn = model.predict(X)

print(f"My own model (denormalized): m = {m_denorm:.5f}, b = {b_denorm:.5f}")
print("Custom Model MSE:", mean_squared_error(y, y_pred_custom))

print("Sklearn Model MSE:", mean_squared_error(y, y_pred_sklearn))
print(f"sklearn model: m is {m_sklearn} and b is {b_sklearn}")




