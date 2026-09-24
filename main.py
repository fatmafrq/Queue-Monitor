import cv2
import csv
from ultralytics import YOLO


# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================

model = YOLO("yolo11s.pt")


# ==========================================
# 2. OPEN VIDEO
# ==========================================

video = cv2.VideoCapture("input/main.mp4")

if not video.isOpened():
    print("Error: Could not open video.")
    exit()


# ==========================================
# 3. QUEUE AREA
# ==========================================

queue_x1 = 250
queue_y1 = 150
queue_x2 = 1700
queue_y2 = 900


# ==========================================
# 4. WAITING TIME VARIABLES
# ==========================================

entry_times = {}
completed_waits = []


# ==========================================
# 5. FINAL STATISTICS VARIABLES
# ==========================================

max_queue_length = 0
total_queue_count = 0
total_frames = 0


# ==========================================
# 6. PROCESS VIDEO
# ==========================================

while True:

    success, frame = video.read()

    if not success:
        break


    # --------------------------------------
    # Current video time
    # --------------------------------------

    video_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000


    # --------------------------------------
    # YOLO TRACKING
    # --------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        conf=0.5,
        iou=0.5,
        verbose=False
    )


    # Set of people currently inside queue
    current_queue_ids = set()

    queue_count = 0


    # --------------------------------------
    # PROCESS DETECTED PEOPLE
    # --------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            # Make sure tracking ID exists
            if box.id is None:
                continue

            person_id = int(box.id[0])

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])


            # ----------------------------------
            # PERSON CENTER
            # ----------------------------------

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # ----------------------------------
            # CHECK IF PERSON IS IN QUEUE
            # ----------------------------------

            inside_queue = (
                queue_x1 < center_x < queue_x2
                and
                queue_y1 < center_y < queue_y2
            )


            # ----------------------------------
            # DRAW PERSON BOX
            # ----------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ----------------------------------
            # PERSON ID
            # ----------------------------------

            cv2.putText(
                frame,
                f"ID: {person_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            # ----------------------------------
            # IF PERSON IS IN QUEUE
            # ----------------------------------

            if inside_queue:

                queue_count += 1

                current_queue_ids.add(person_id)


                # ----------------------------------
                # RECORD ENTRY TIME
                # ----------------------------------

                if person_id not in entry_times:

                    entry_times[person_id] = video_time


                # ----------------------------------
                # CALCULATE WAITING TIME
                # ----------------------------------

                waiting_time = (
                    video_time - entry_times[person_id]
                )


                # ----------------------------------
                # DISPLAY WAITING TIME
                # ----------------------------------

                cv2.putText(
                    frame,
                    f"Wait: {waiting_time:.1f}s",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )


    # ==========================================
    # 7. DETECT PEOPLE WHO LEFT THE QUEUE
    # ==========================================

    for person_id in list(entry_times.keys()):

        if person_id not in current_queue_ids:

            waiting_time = (
                video_time - entry_times[person_id]
            )

            completed_waits.append(waiting_time)

            del entry_times[person_id]


    # ==========================================
    # 8. UPDATE QUEUE STATISTICS
    # ==========================================

    total_queue_count += queue_count
    total_frames += 1


    if queue_count > max_queue_length:

        max_queue_length = queue_count


    # ==========================================
    # 9. DRAW QUEUE AREA
    # ==========================================

    cv2.rectangle(
        frame,
        (queue_x1, queue_y1),
        (queue_x2, queue_y2),
        (255, 0, 0),
        3
    )


    cv2.putText(
        frame,
        "QUEUE AREA",
        (queue_x1, queue_y1 - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3
    )


    # ==========================================
    # 10. CURRENT AVERAGE WAITING TIME
    # ==========================================

    if len(completed_waits) > 0:

        current_average_wait = (
            sum(completed_waits) /
            len(completed_waits)
        )

    else:

        current_average_wait = 0


    # ==========================================
    # 11. LIVE DASHBOARD
    # ==========================================

    cv2.putText(
        frame,
        f"Queue Length: {queue_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Max Queue: {max_queue_length}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Avg Wait: {current_average_wait:.1f} sec",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


    # ==========================================
    # 12. RESIZE FOR DISPLAY
    # ==========================================

    display_frame = cv2.resize(
        frame,
        (960, 540)
    )


    cv2.imshow(
        "Queue Length & Waiting Time Monitor",
        display_frame
    )


    # ==========================================
    # 13. PRESS Q TO STOP
    # ==========================================

    if cv2.waitKey(25) & 0xFF == ord("q"):

        break


# ==========================================
# 14. FINAL RESULTS
# ==========================================

if total_frames > 0:

    average_queue_length = (
        total_queue_count /
        total_frames
    )

else:

    average_queue_length = 0


if len(completed_waits) > 0:

    final_average_wait = (
        sum(completed_waits) /
        len(completed_waits)
    )

else:

    final_average_wait = 0


# ==========================================
# 15. RELEASE VIDEO
# ==========================================

video.release()

cv2.destroyAllWindows()


# ==========================================
# 16. PRINT FINAL RESULTS
# ==========================================

print("\n========== FINAL RESULTS ==========")

print(
    f"Maximum Queue Length: "
    f"{max_queue_length}"
)

print(
    f"Average Queue Length: "
    f"{average_queue_length:.2f}"
)

print(
    f"Average Waiting Time: "
    f"{final_average_wait:.2f} seconds"
)

print(
    f"People Processed: "
    f"{len(completed_waits)}"
)

print("===================================")


# ==========================================
# 17. SAVE RESULTS TO CSV
# ==========================================

with open(
    "queue_results.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)


    writer.writerow([
        "Maximum Queue Length",
        "Average Queue Length",
        "Average Waiting Time (seconds)",
        "People Processed"
    ])


    writer.writerow([
        max_queue_length,
        round(average_queue_length, 2),
        round(final_average_wait, 2),
        len(completed_waits)
    ])


print("Results saved to queue_results.csv")
