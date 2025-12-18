import time
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------
# STEP 1: Train ML Model (Demo Data)
# ----------------------------------

# Training dataset (replace with real sensor data)
X_train = np.array([
    [10, 7.2, 120, 0],   # Clean
    [15, 7.0, 150, 0],
    [70, 6.0, 600, 1],   # Contaminated
    [80, 5.8, 700, 1],
    [20, 7.5, 200, 0],
    [90, 6.2, 800, 1],
    [35, 8.0, 300, 0],
    [60, 6.1, 650, 1]
])

y_train = np.array([0, 0, 1, 1, 0, 1, 0, 1])  # 0=Clean, 1=Contaminated

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("ML Model Trained Successfully\n")

# ----------------------------------
# STEP 2: Sensor Reading Functions
# (Replace with actual sensor code)
# ----------------------------------

def read_turbidity():
    return random.uniform(0, 100)   # NTU

def read_ph():
    return random.uniform(5.5, 9.0)

def read_tds():
    return random.uniform(50, 800)  # ppm

def read_garbage_sensor():
    return random.choice([0, 1])    # 1 = garbage detected

# ----------------------------------
# STEP 3: False Alarm Reduction
# ----------------------------------

CONFIRMATION_COUNT = 3
alert_counter = 0

# ----------------------------------
# STEP 4: Real-Time Detection Loop
# ----------------------------------

while True:
    turbidity = read_turbidity()
    ph = read_ph()
    tds = read_tds()
    garbage = read_garbage_sensor()

    sensor_data = np.array([[turbidity, ph, tds, garbage]])
    prediction = model.predict(sensor_data)

    print(f"Turbidity: {turbidity:.2f} | pH: {ph:.2f} | TDS: {tds:.2f} | Garbage: {garbage}")

    if prediction[0] == 1:
        alert_counter += 1
    else:
        alert_counter = 0

    if alert_counter >= CONFIRMATION_COUNT:
        print("⚠️ ALERT: Water Contamination Detected!")
        print("🧹 Cleaning system activated")
        print("📱 Notification sent to monitoring device\n")
        alert_counter = 0

    time.sleep(2)

