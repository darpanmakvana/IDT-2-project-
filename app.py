# import streamlit as st
# import pandas as pd
# import time 
# from datetime import datetime

# ts=time.time()
# date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
# timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")

# from streamlit_autorefresh import st_autorefresh

# count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

# if count == 0:
#     st.write("Count is zero")
# elif count % 3 == 0 and count % 5 == 0:
#     st.write("FizzBuzz")
# elif count % 3 == 0:
#     st.write("Fizz")
# elif count % 5 == 0:
#     st.write("Buzz")
# else:
#     st.write(f"Count: {count}")


# date = datetime.now().strftime("%d-%m-%Y")
# file_path = f"Attendance/Attendance_{date}.csv"


# st.dataframe(df.style.highlight_max(axis=0))


import streamlit as st
import pandas as pd
import time 
from datetime import datetime
import os
from streamlit_autorefresh import st_autorefresh

# Get the current timestamp and date
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

# Auto-refresh every 2 seconds
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

# Display FizzBuzz logic based on the count
if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

# Define the file path for today's attendance CSV
file_path = f"/home/uday/Desktop/face_recognition_project-main/Attendance/Attendance_{date}.csv"

# Check if the file exists and read it, else create a new one
if not os.path.exists(file_path):
    st.write(f"File not found: {file_path}. Creating a new empty file.")
    # Create an empty DataFrame with default attendance columns
    df = pd.DataFrame(columns=['Name', 'Time', 'Status'])
    # Save the empty DataFrame as a new CSV file
    df.to_csv(file_path, index=False)
else:
    df = pd.read_csv(file_path)

# Display the DataFrame (whether loaded or newly created)
st.dataframe(df.style.highlight_max(axis=0))
