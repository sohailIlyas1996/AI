from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

X = [
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 0],
    [4, 1],
    [5, 0],
    [5, 1],
    [6, 1],
    [7, 0],
    [8, 1],
]

y = [
    "Fail",
    "Fail",
    "Fail",
    "Fail",
    "Pass",
    "Fail",
    "Pass",
    "Pass",
    "Pass",
    "Pass",
]

new_student = [[6, 1]]


decision_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

decision_tree.fit(X, y)

tree_prediction = decision_tree.predict(new_student)
print("Decision tree prediction:", tree_prediction[0])

# Display the questions learned by the tree.
print("\nDecision tree rules:")
print(
    export_text(
        decision_tree,
        feature_names=["study_hours", "practice_completed"]
    )
)


random_forest = RandomForestClassifier(
    n_estimators=5,
    max_depth=3,
    random_state=42
)

random_forest.fit(X, y)

forest_prediction = random_forest.predict(new_student)
print("Random forest prediction:", forest_prediction[0])

print("Number of trees:", len(random_forest.estimators_))