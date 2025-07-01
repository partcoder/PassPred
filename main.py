import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



def main():
    subject = input("Choose subject (maths/science/english): ").strip().lower()
    if subject == "maths" or subject == "math":
        filename = 'maths.csv'
    elif subject == "science":
        filename = 'science.csv'
    elif subject == "english":
        filename = 'english.csv'
    else:
        print("Invalid subject.")
        exit()

    df = pd.read_csv(filename)
    X = df[['study_hours', 'days_before_exam', 'attendance_percentage']]
    y = df['pass']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Take input from user
    study_hours = float(input("Enter study hours: "))
    days_before_exam = float(input("Enter days before exam: "))
    attendance_percentage = float(input("Enter attendance percentage: "))

    new_student = pd.DataFrame(
        [[study_hours, days_before_exam, attendance_percentage]],
        columns=['study_hours', 'days_before_exam', 'attendance_percentage']
    )
    prediction = model.predict(new_student)
    print("Will the student pass?", "Yes" if prediction[0] == 1 else "No")

while True:
    main()
