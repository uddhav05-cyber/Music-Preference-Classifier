import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Generate synthetic dataset
np.random.seed(42)
data = {
    'age': np.random.randint(10, 70, 500),
    'gender': np.random.choice([0, 1], 500),  # 0: Female, 1: Male
}

# Create music preferences based on patterns
music_preferences = []
for age, gender in zip(data['age'], data['gender']):
    if age < 25:
        pref = 'Pop' if gender == 0 else 'Hip-Hop'
    elif age < 40:
        pref = 'Rock' if gender == 1 else 'Pop'
    elif age < 55:
        pref = 'Jazz' if gender == 0 else 'Classical'
    else:
        pref = 'Classical'
    music_preferences.append(pref)

data['music_preference'] = music_preferences

# Create DataFrame
df = pd.DataFrame(data)
df.to_csv('music_data.csv', index=False)

# Train model
X = df[['age', 'gender']]
y = df['music_preference']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Save model
with open('music_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")