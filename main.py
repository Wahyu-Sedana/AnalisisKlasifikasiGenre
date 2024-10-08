import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

X = np.random.rand(100, 10)
y = np.random.choice(['Pop', 'Reggae', 'Disco'], 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)

start_time = time.time()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end_time = time.time()

running_time = end_time - start_time

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

genres = list(report.keys())[:-3]
running_times = [round(running_time, 3)] * len(genres)
total_lagu = [len(y_test[y_test == genre]) for genre in genres]
prediksi_benar = [int(report[genre]['support'] * report[genre]['recall']) for genre in genres]

data = {
    "No": list(range(1, len(genres) + 1)),
    "Genre": genres,
    "Running time": running_times,
    "Total lagu": total_lagu,
    "Prediksi Benar": prediksi_benar
}

df = pd.DataFrame(data)
print(df)

output_file = 'hasil_prediksi_genre.xlsx'
df.to_excel(output_file, index=False)
print(f'Hasil prediksi telah disimpan di {output_file}')
