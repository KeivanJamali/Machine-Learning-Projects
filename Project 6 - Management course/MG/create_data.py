import numpy
import numpy as np
import pandas
import os
from pathlib import Path
import random
import datetime
from sklearn.preprocessing import StandardScaler
import pandas as pd


def pattern(x):
    return (5 * x ** 2 + 10 * x + 100)


def pattern_dur(x):
    return (+5 * x ** 2 - 12 * x + 100)


######################################################################################################################
number_of_data = 10000
######################################################################################################################
keywords = ['International', 'Global', 'World', 'National', 'Annual', 'International Workshop', 'Symposium',
            'Conference', 'Congress', 'Summit', 'Forum', 'Global Summit', 'International Congress',
            'National Conference', 'Annual Symposium', 'World Forum', 'International Workshop',
            'Virtual Event', 'Industry Expo', 'Academic Seminar', 'Expert Panel', 'Technical Colloquium',
            'Thought Leadership Retreat', 'Hackathon']

topics = ['Artificial Intelligence', 'Machine Learning', 'Data Science', 'Big Data', 'Cloud Computing', 'Cybersecurity',
          'Internet of Things', 'Robotics', 'Natural Language Processing', 'Computer Vision', 'Blockchain',
          'Software Engineering', 'Computer Graphics', 'Virtual Reality', 'Augmented Reality', 'Data Mining',
          'Deep Learning', 'High-Performance Computing', 'Information Retrieval', 'Quantum Computing', 'Cryptography',
          'Internet of Things Applications', 'Text Mining', 'Human-Computer Interaction',
          'Privacy-Preserving Technologies', 'Recommender Systems', 'Computational Biology', 'Bioinformatics',
          'Smart Cities', 'Urban Analytics', 'Explainable AI', 'Healthcare', 'Machine Vision', 'Explainable AI',
          'Edge Computing', 'Reinforcement Learning', 'Natural Language Understanding', 'Predictive Analytics',
          'Autonomous Systems', 'Internet of Medical Things', 'Federated Learning', 'Cognitive Computing',
          'Social Network Analysis', 'Computer-Assisted Diagnosis', 'Genetic Algorithms', 'Quantum Machine Learning',
          'Network Security', 'Computer-Assisted Language Learning', 'Intelligent Transportation Systems',
          'Speech Recognition', 'Computer-Aided Drug Design', 'Sports Analytics', 'Data Visualization',
          'Knowledge Graphs', 'Sentiment Analysis', 'Precision Medicine', 'Computational Neuroscience',
          'Swarm Intelligence', 'Explainable Robotics', 'Human-Robot Interaction', 'Healthcare Informatics']

conference_names = []

while len(conference_names) < number_of_data:
    keyword = random.choice(keywords)
    topic = random.choice(topics)
    conference = f"{keyword} {topic}"
    # if conference not in conference_names:
    conference_names.append(conference)
print(len(conference_names))
######################################################################################################################
hall = ["Jaber", "Theater", "Mechanic", "Kahroba", "Rabiee", "Sabz", "Borgeii", "Jabari"]
quality_of_sound_rate = [5, 2, 5, 3, 4, 1, 2, 1]
capacity = [280, 150, 150, 100, 300, 120, 90, 1000]
rent = [20, 18, 30, 120, 25, 10, 12, 10]
halls = [random.choice(hall) for _ in range(number_of_data)]
quality_of_sound = []
capacity_of_halls = []
renting = []
for i in halls:
    quality_of_sound.append(quality_of_sound_rate[hall.index(i)])
    capacity_of_halls.append(capacity[hall.index(i)])
    renting.append(rent[hall.index(i)])

print(len(quality_of_sound), len(capacity_of_halls), len(renting))
######################################################################################################################
current_date = datetime.date.today()
start_date = current_date - datetime.timedelta(days=365 * 30)
end_date = current_date
dates = []
for _ in range(number_of_data):
    random_date = start_date + datetime.timedelta(days=random.randint(0, (end_date - start_date).days))
    dates.append(random_date)
dates = sorted(dates)
print(len(dates))
######################################################################################################################
min_duration = 30
max_duration = 300  # 5 hours = 300 minutes

durations = []
for i in range(number_of_data):
    duration_minutes = pattern_dur(i)
    durations.append(duration_minutes)
print(len(durations))
######################################################################################################################
# Generate random numbers within the range
number_of_attendance = []
for i in capacity_of_halls:
    number_of_attendance.append(random.randint(i - 20, i))
print(len(number_of_attendance))
######################################################################################################################
international = []
for i in range(number_of_data):
    if i < number_of_data // 2:
        international.append(False)
    else:
        international.append(True)
print(len(international))
######################################################################################################################
expanses = []
for i in range(number_of_data):
    if i < 5000:
        expanses.append(pattern(i))
    elif i < 7000:
        expanses.append(pattern(i))
    elif i < 8000:
        expanses.append(pattern(i))
    elif i < 9000:
        expanses.append(pattern(i))
    elif i < 10000:
        expanses.append(pattern(i))

print(len(expanses))
######################################################################################################################

scaler = StandardScaler()
durations = scaler.fit_transform(np.array(durations).reshape(-1, 1)).squeeze()
expanses = scaler.fit_transform(np.array(expanses).reshape(-1, 1)).squeeze()
number_of_attendance = scaler.fit_transform(np.array(number_of_attendance).reshape(-1, 1)).squeeze()

data = pd.DataFrame({"Event_name": conference_names,
                     "Date": dates,
                     "Duration": durations,
                     "Number_of_Attendance": number_of_attendance,
                     "International": international,
                     "Expenses": expanses,
                     "Place": halls,
                     "Renting": renting,
                     "Capacity": capacity_of_halls,
                     "Sound_System_quality": quality_of_sound,
                     })

data.to_csv("conference_data.csv", index=False)
