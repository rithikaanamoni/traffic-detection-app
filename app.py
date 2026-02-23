import os
os.environ["STREAMLIT_WATCHDOG_DISABLE"] = "true"  # Disable folder watch to avoid inotify errors

import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Traffic Density Analysis", layout="wide")
st.title("🚦 Traffic Density Analysis System")

uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(tfile.name)
    st.info("Processing video... This may take some time ⏳")

    try:
        model = YOLO("best.pt")  # Load your trained YOLO model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counts = []

    progress_bar = st.progress(0)
    frame_idx = 0
    frame_skip = 2  # Process every 2nd frame to speed up

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(frame_rgb, verbose=False)
            count = sum(len(r.boxes) for r in results)
            frame_counts.append(count)

        frame_idx += 1
        progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    st.success("✅ Processing Completed!")

    # Segment traffic analysis
    segment_duration = 5  # seconds
    frames_per_segment = int(fps * segment_duration / frame_skip)
    segments = []
    avg_counts = []

    for i in range(0, len(frame_counts), frames_per_segment):
        segment = frame_counts[i:i + frames_per_segment]
        if len(segment) == 0:
            continue

        avg_counts.append(np.mean(segment))
        start_time = i * frame_skip / fps
        end_time = (i + len(segment)) * frame_skip / fps

        segments.append(f"{int(start_time // 60):02}:{int(start_time % 60):02} - "
                        f"{int(end_time // 60):02}:{int(end_time % 60):02}")

    df = pd.DataFrame({
        "time_hms": segments,
        "avg_vehicle_per_frame": avg_counts
    })

    # Traffic level summary
    overall_avg = np.mean(avg_counts)
    if overall_avg < 5:
        level = "Low Traffic"
    elif overall_avg < 15:
        level = "Moderate Traffic"
    else:
        level = "High Traffic"

    peak_index = np.argmax(avg_counts)
    peak_time = segments[peak_index]
    peak_value = avg_counts[peak_index]

    # Display summary
    st.subheader("📊 Traffic Summary")
    st.write(f"**Traffic Level:** {level}")
    st.write(f"**Peak Traffic Time:** {peak_time}")
    st.write(f"**Peak Vehicle Average:** {round(peak_value, 2)}")

    # Plot traffic trend
    st.subheader("📈 Traffic Trend Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["time_hms"], df["avg_vehicle_per_frame"], marker='o', color='blue')
    ax.set_xlabel("Time Segment")
    ax.set_ylabel("Avg Vehicles per Frame")
    ax.set_title("Traffic Density Over Time")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # CSV download
    st.subheader("💾 Download Traffic Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="traffic_analysis.csv",
        mime="text/csv"
    )
