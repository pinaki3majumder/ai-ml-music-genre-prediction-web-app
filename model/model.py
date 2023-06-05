import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as rf
import joblib
from pathlib import Path
import os


load_path = Path(__file__).parent
file_name = "/music.csv"

# import csv file
music_data = pd.read_csv(f"{load_path}{file_name}")

# table without 'genre' column
user_details = music_data.drop(columns=['genre'])

# table with only 'genre' column
music_genre = music_data['genre']

dump_path = os.path.split(load_path)
pkl_file_name = "\model.pkl"

# create .pkl file with joblib
clf = rf()
clf.fit(user_details.values, music_genre.values)
joblib.dump(clf, f"{dump_path[0]}{pkl_file_name}")

# prediction model
model = DecisionTreeClassifier()
# fix warning with https://stackoverflow.com/a/70278753/3693763
model.fit(user_details.values, music_genre.values)
prediction = model.predict([[22, 1]])
print(prediction)
