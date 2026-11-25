from sklearn.linear_model import LogisticRegression


X = [[1], [2], [3], [4], [5], [6], [7], [8]]

y = [0, 0, 0, 0, 1, 1, 1, 1]

model = LogisticRegression()
model.fit(X, y)

hours = [[6]]
prediction = model.predict(hours)[0]
pass_probability = model.predict_proba(hours)[0, 1]

print(f"Hours studied: {hours[0][0]}")
print("Prediction:", "Pass" if prediction == 1 else "Fail")
print(f"Estimated probability of passing: {pass_probability:.1%}")