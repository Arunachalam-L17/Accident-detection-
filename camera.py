import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
from email.message import EmailMessage
import ssl
import smtplib

# Load accident detection model
model = AccidentDetectionModel("model.json", 'model_weights.keras')
font = cv2.FONT_HERSHEY_SIMPLEX

# Email credentials and details
email_sender = "k.r.deepasri16@gmail.com"
email_password = "nkzy pdlb etec ubbg"  # Use an app password for security
email_receiver = "deepasrikailasam@gmail.com"

subject = "Accident Alert"
body = """
An accident has been detected at this location. Please investigate immediately.
"""


# Function to send email
def send_email():
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

    # Secure SSL context
    context = ssl.create_default_context()

    # Sending the email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())


# Function to start video application
def startapplication():
    video = cv2.VideoCapture('aci.mp4')  # For camera use video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Error: Video not found or can't be opened.")
        return

    email_sent = False  # Flag to prevent sending multiple emails

    while True:
        ret, frame = video.read()

        if not ret:
            print("End of video or can't read frame.")
            break

        # Convert frame to RGB and resize for the model
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        # Predict accident and get probability
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])

        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)

            # Display accident probability on the video
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred + " " + str(prob) + "%", (20, 30), font, 1, (255, 255, 0), 2)

            # If accident probability hits 100% and email has not been sent
            if prob == 100 and not email_sent:
                send_email()
                email_sent = True  # Set the flag to True after sending the email

        # Show the video frame with predictions
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    startapplication()
