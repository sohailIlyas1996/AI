from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]

y = [40, 50, 60, 70, 80]

model = LinearRegression()
model.fit(X, y)

hours = [[10]]
predicted_mark = model.predict(hours)[0]

print(f"Hours studied: {hours[0][0]}")
print(f"Predicted exam mark: {predicted_mark:.1f}")