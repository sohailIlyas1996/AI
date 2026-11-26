from sklearn.neighbors import KNeighborsRegressor

sizes = [[60], [80], [95], [105], [120], [200]]
prices = [150, 180, 200, 220, 240, 400]

model = KNeighborsRegressor(n_neighbors=3)
model.fit(sizes, prices)

print(model.predict([[100]]))