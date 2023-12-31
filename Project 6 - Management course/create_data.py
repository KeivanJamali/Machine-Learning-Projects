import numpy
import pandas
import os
from pathlib import Path
import random
import datetime

import pandas as pd

######################################################################################################################
number_of_data = 1000
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
    if conference not in conference_names:
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
start_date = current_date - datetime.timedelta(days=365 * 5)
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

# Generate random durations within the range
durations = []
for _ in range(number_of_data):
    duration_minutes = random.randint(min_duration, max_duration)
    durations.append(duration_minutes)
print(len(durations))
######################################################################################################################
min_number = 10
# Generate random numbers within the range
number_of_attendance = []
for i in capacity_of_halls:
    number_of_attendance.append(random.randint(min_number, i))
print(len(number_of_attendance))
######################################################################################################################
international = [random.choice([True, False]) for _ in range(number_of_data)]
print(len(international))
######################################################################################################################
min_expense = 10
max_expense = 1000
expanses = []
expanses.extend(random.randint(10, 30) for _ in range(500))
expanses.extend(random.randint(50, 110) for _ in range(200))
expanses.extend(random.randint(150, 200) for _ in range(100))
expanses.extend(random.randint(300, 700) for _ in range(100))
expanses.extend(random.randint(700, 1500) for _ in range(100))

random.shuffle(expanses)
print(len(expanses))
######################################################################################################################

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
