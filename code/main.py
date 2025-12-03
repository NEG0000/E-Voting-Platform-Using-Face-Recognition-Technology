from flask import Flask, render_template, url_for, request, session, flash, redirect
from flask_mail import *
from email.mime.multipart import MIMEMultipart
from werkzeug.utils import secure_filename
import smtplib
import pymysql
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
import shutil
import datetime
import time
from sklearn import preprocessing
import requests
facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)
if cascade.empty():
    print("Failed to load Haar cascade.")
    exit()

mydb=pymysql.connect(host='localhost', user='root', password='', port=3306, database='smart_voting_system')

sender_address = 'id@gmail.com' #enter sender's email id
sender_pass = 'pass' #enter sender's password

app=Flask(__name__)
app.config['SECRET_KEY']='key'

# 🧠 Prevent caching of pages (important for logout)
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.before_request
def set_session_defaults():
    if 'IsAdmin' not in session:
        session['IsAdmin'] = False
    if 'User' not in session:
        session['User'] = None

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/admin', methods=['POST', 'GET'])
def admin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if email == 'admin@voting.com' and password == 'admin':
            session['IsAdmin'] = True
            session['User'] = 'admin'
            flash('Admin login successful', 'success')
        else:
            flash('Invalid credentials', 'danger')
            return redirect(url_for('admin'))

    return render_template('admin.html', admin=session.get('IsAdmin'))

@app.before_request
def check_session_timeout():
    # Set session timeout duration in seconds
    session_timeout = 3600  # 1 hour

    # Check if the user is logged in and if the session is expired
    if session.get('IsAdmin'):
        last_activity_time = session.get('last_activity_time')
        current_time = time.time()

        # If session expired, log out the user
        if last_activity_time and current_time - last_activity_time > session_timeout:
            session.pop('IsAdmin', None)
            session.pop('User', None)
            flash('Your session has expired. Please log in again.', 'warning')

        # Update last activity time
        session['last_activity_time'] = current_time

@app.route('/logout')
def logout():
    session.clear()  # This clears the entire session
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('home'))  # Redirect to the admin login page


@app.route('/add_nominee', methods=['POST','GET'])
def add_nominee():
    if request.method=='POST':
        member=request.form['member_name']
        party=request.form['party_name']
        logo=request.form['test']
        nominee=pd.read_sql_query('SELECT * FROM nominee', mydb)
        all_members=nominee.member_name.values
        all_parties=nominee.party_name.values
        all_symbols=nominee.symbol_name.values
        if member in all_members:
            flash(r'The member already exists', 'info')
        elif party in all_parties:
            flash(r"The party already exists", 'info')
        elif logo in all_symbols:
            flash(r"The logo is already taken", 'info')
        else:
            sql="INSERT INTO nominee (member_name, party_name, symbol_name) VALUES (%s, %s, %s)"
            cur=mydb.cursor()
            cur.execute(sql, (member, party, logo))
            mydb.commit()
            cur.close()
            flash(r"Successfully registered a new nominee", 'primary')
    return render_template('nominee.html', admin=session['IsAdmin'])

def delete_voter_from_db_and_sync(id):
    try:
        cursor = mydb.cursor()

        # First, check if the ID is Aadhar or Voter ID
        cursor.execute("SELECT aadhar_id FROM voters WHERE aadhar_id = %s", (id,))
        result = cursor.fetchone()

        if result:  # If Aadhar ID exists in database
            # Delete record from the database
            cursor.execute("DELETE FROM voters WHERE aadhar_id = %s", (id,))
            mydb.commit()
            flash(f"Voter with Aadhar ID {id} has been deleted.", "success")
            
            # Remove corresponding face images from the dataset folder
            person_folder = os.path.join("dataset", id)
            if os.path.exists(person_folder):
                shutil.rmtree(person_folder)
                print(f"Deleted images for Aadhar ID {id}")
        
        else:
            cursor.execute("SELECT voter_id FROM voters WHERE voter_id = %s", (id,))
            result = cursor.fetchone()

            if result:  # If Voter ID exists in database
                # Delete record from the database
                cursor.execute("DELETE FROM voters WHERE voter_id = %s", (id,))
                mydb.commit()
                flash(f"Voter with Voter ID {id} has been deleted.", "success")
                
                # Remove corresponding face images from the dataset folder
                person_folder = os.path.join("dataset", id)
                if os.path.exists(person_folder):
                    shutil.rmtree(person_folder)
                    print(f"Deleted images for Voter ID {id}")
            else:
                flash("No matching Voter or Aadhar ID found.", "danger")
                return redirect(url_for('admin_dashboard'))

        # ✅ Retrain model after deletion
        retrain_model()

        return redirect(url_for('home'))

    except Exception as e:
        print(f"Error deleting voter: {e}")
        flash("An error occurred while deleting the voter.", "danger")
        return redirect(url_for('admin_dashboard'))

@app.route('/registration', methods=['POST','GET'])
def registration():
    if request.method=='POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        state = request.form['state']
        d_name = request.form['d_name']

        middle_name = request.form['middle_name']
        aadhar_id = request.form['aadhar_id']
        voter_id = request.form['voter_id']
        pno = request.form['pno']
        age = int(request.form['age'])
        email = request.form['email']
        voters=pd.read_sql_query('SELECT * FROM voters', mydb)
        all_aadhar_ids=voters.aadhar_id.values
        all_voter_ids=voters.voter_id.values
        if age >= 18:
            if (aadhar_id in all_aadhar_ids) or (voter_id in all_voter_ids):
                flash(r'Already Registered as a Voter')
            else:
                sql = 'INSERT INTO voters (first_name, middle_name, last_name, aadhar_id, voter_id, email,pno,state,d_name, verified) VALUES (%s,%s,%s, %s, %s, %s, %s, %s, %s, %s)'
                cur=mydb.cursor()
                cur.execute(sql, (first_name, middle_name, last_name, aadhar_id, voter_id, email, pno, state, d_name, 'no'))
                mydb.commit()
                cur.close()
                session['aadhar']=aadhar_id
                session['status']='no'
                session['email']=email
                return redirect(url_for('verify'))
        else:
            flash("if age less than 18 than not eligible for voting","info")

            delete_voter_from_db_and_sync(id)


    return render_template('voter_reg.html')

@app.route('/verify', methods=['POST','GET'])
def verify():
    if session['status']=='no':
        if request.method=='POST':
            otp_check=request.form['otp_check']
            if otp_check==session['otp']:
                session['status']='yes'
                sql="UPDATE voters SET verified='%s' WHERE aadhar_id='%s'"%(session['status'], session['aadhar'])
                cur=mydb.cursor()
                cur.execute(sql)
                mydb.commit()
                cur.close()
                flash(r"Email verified successfully",'primary')
                return redirect(url_for('capture_images')) #change it to capture photos
            else:
                flash(r"Wrong OTP. Please try again.","info")
                return redirect(url_for('verify'))
        else:
            #Sending OTP
            message = MIMEMultipart()
            receiver_address = session['email']
            message['From'] = sender_address
            message['To'] = receiver_address
            Otp = str(np.random.randint(100000, 999999))
            session['otp']=Otp
            message.attach(MIMEText(session['otp'], 'plain'))
            abc = smtplib.SMTP('smtp.gmail.com', 587)
            abc.starttls()
            abc.login(sender_address, sender_pass)
            text = message.as_string()
            abc.sendmail(sender_address, receiver_address, text)
            abc.quit()
    else:
        flash(r"Your email is already verified", 'warning')
    return render_template('verify.html')

def retrain_model():
    # Path to dataset
    dataset_path = "dataset"
    
    # Initialize face detector and recognizer
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Prepare data and labels
    faces = []
    labels = []

    # Loop through dataset and exclude the deleted voter (by Aadhaar ID)
    deleted_aadhaar_id = 'some_aadhar_id'  # Replace with the Aadhaar ID that was deleted

    for aadhar_id in os.listdir(dataset_path):
        if aadhar_id == deleted_aadhaar_id:  # Skip the deleted Aadhaar ID
            continue
        
        person_dir = os.path.join(dataset_path, aadhar_id)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in detected_faces:
                faces.append(gray[y:y + h, x:x + w])
                labels.append(aadhar_id)

    # Check if faces and labels are populated
    if not faces or not labels:
        raise ValueError("No face data found. Please check the dataset.")

    # Encode labels
    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # Train the recognizer
    recognizer.train(faces, np.array(encoded_labels))

    # Save the retrained model
    recognizer.save("Trained.yml")

    # Save the label encoder
    with open("encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Model retraining complete. Files updated: Trained.yml and encoder.pkl.")

@app.route('/capture_images', methods=['POST', 'GET'])
def capture_images():
    if request.method == 'POST':
        # Initialize camera
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cam.isOpened():
            print("❌ Failed to open camera.")
            flash("Failed to open camera. Please check your device.", "danger")
            return redirect(url_for('home'))
        
        sampleNum = 0
        # Get path to store images based on Aadhaar ID in session
        aadhar_id = session.get('aadhar', None)
        if not aadhar_id:
            flash("Aadhaar ID not found. Please log in again.", "danger")
            return redirect(url_for('home'))

        # Store the images in the "dataset" folder
        path_to_store = os.path.join(os.getcwd(), "dataset", aadhar_id)
        
        # Attempt to clear existing directory before creating a new one
        try:
            shutil.rmtree(path_to_store)
        except FileNotFoundError:
            pass  # It's okay if the folder doesn't exist
        os.makedirs(path_to_store, exist_ok=True)

        # Start capturing images
        while True:
            ret, img = cam.read()
            if not ret:
                print("❌ Failed to capture image.")
                flash("Failed to capture image. Please try again.", "danger")
                break

            # Flip the image horizontally for mirror effect
            img = cv2.flip(img, 1)

            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Error converting to grayscale: {e}")
                flash("Error processing image. Please try again.", "danger")
                continue

            # Use cascade to detect faces
            faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Increment sample number
                sampleNum += 1
                
                # Save the captured face image to dataset folder
                face_image_path = os.path.join(path_to_store, f"{sampleNum}.jpg")
                cv2.imwrite(face_image_path, gray[y:y + h, x:x + w])

                # Display face sample number
                cv2.putText(img, f"Face {sampleNum}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Display the image with the face rectangle
            cv2.imshow('frame', img)
            cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)

            # Wait for 100ms, and break if the user presses 'q'
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

            # Stop capturing after 200 images
            if sampleNum >= 30:
                break

        # Release the camera and close OpenCV windows
        cam.release()
        cv2.destroyAllWindows()

        

        # Retrain model after capturing images
        retrain_model()

        flash("Registration is successful and model is retrained.", "success")
        return redirect(url_for('home'))

    return render_template('capture.html')

from sklearn.preprocessing import LabelEncoder
import pickle
le = LabelEncoder()

def getImagesAndLabels(path):
    folderPaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    global le
    for folder in folderPaths:
        imagePaths = [os.path.join(folder, f) for f in os.listdir(folder)]
        aadhar_id = folder.split("\\")[1]
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(aadhar_id)
            # Ids.append(int(aadhar_id))
    Ids_new=le.fit_transform(Ids).tolist()
    output = open('encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    return faces, Ids_new

@app.route('/train', methods=['POST','GET'])
def train():
    if request.method=='POST':
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, Id = getImagesAndLabels(r"dataset")
        print(Id)
        print(len(Id))
        recognizer.train(faces, np.array(Id))
        recognizer.save("Trained.yml")
        flash(r"Model Trained Successfully", 'Primary')
        return redirect(url_for('home'))
    return render_template('train.html')
@app.route('/update')
def update():
    return render_template('update.html')
@app.route('/updateback', methods=['POST','GET'])
def updateback():
    if request.method == 'POST':
        # Retrieving data from the form
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        middle_name = request.form['middle_name']
        aadhar_id = request.form['aadhar_id']
        voter_id = request.form['voter_id']
        email = request.form['email']
        pno = request.form['pno']
        age = int(request.form['age'])

        # Query the database for existing voters
        try:
            voters = pd.read_sql_query('SELECT * FROM voters', mydb)
            all_aadhar_ids = voters.aadhar_id.values

            # Check age eligibility
            if age >= 18:
                # Check if the Aadhar ID exists in the database
                if aadhar_id in all_aadhar_ids:
                    # Prepare the SQL update query
                    sql = """
                        UPDATE VOTERS 
                        SET first_name=%s, middle_name=%s, last_name=%s, voter_id=%s, email=%s, pno=%s, verified=%s 
                        WHERE aadhar_id=%s
                    """
                    # Execute the query
                    cur = mydb.cursor()
                    cur.execute(sql, (first_name, middle_name, last_name, voter_id, email, pno, 'no', aadhar_id))
                    mydb.commit()
                    cur.close()

                    # Update session variables
                    session['aadhar'] = aadhar_id
                    session['status'] = 'no'
                    session['email'] = email

                    flash('Database Updated Successfully', 'Primary')
                    return redirect(url_for('verify'))
                else:
                    flash(f"Aadhar: {aadhar_id} doesn't exist in the database for updation", 'warning')
            else:
                flash("Age must be 18 or greater to be eligible", "info")
        except Exception as e:
            flash(f"An error occurred: {e}", "danger")

    return render_template('update.html')

@app.route('/voting', methods=['POST', 'GET'])
def voting():
    if request.method == 'POST':
        try:
            # Check if encoder file exists
            if not os.path.exists("encoder.pkl"):
                flash("Encoder file not found. Please train the system first.", "danger")
                return redirect(url_for('home'))

            # Load Label Encoder
            with open('encoder.pkl', 'rb') as pkl_file:
                my_le = pickle.load(pkl_file)

            # Load Face Recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(r"C:\Users\Fahim\Downloads\Smart-Voting-System-main\Smart-Voting-System-main\code\Trained.yml")

            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            font = cv2.FONT_HERSHEY_SIMPLEX

            match_counts = {}
            frame_count = 0
            max_frames = 200
            required_matches = 20

            while frame_count < max_frames:
                ret, im = cam.read()
                if not ret:
                    flash("Unable to capture frame. Please check the camera.", "danger")
                    break
                im = cv2.flip(im, 1)
                frame_count += 1

                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.2, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)

                    Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                    print(f"[INFO] Recognized ID: {Id}, Confidence: {conf}")

                    if conf > 30  :  # Confidence threshold adjusted
                        try:
                            det_aadhar = my_le.inverse_transform([Id])[0]
                            print(f"[INFO] Aadhaar detected: {det_aadhar}")

                            # Check if Aadhaar exists in DB
                            cursor = mydb.cursor()
                            cursor.execute("SELECT aadhar_id FROM voters WHERE aadhar_id = %s", (det_aadhar,))
                            result = cursor.fetchone()

                            if result:
                                match_counts[det_aadhar] = match_counts.get(det_aadhar, 0) + 1

                                # Display Aadhaar
                                cv2.putText(im, f"Aadhar: {det_aadhar}", (x, y - 10), font, 0.8, (255, 255, 255), 2)

                                # Progress bar
                                progress = min(match_counts[det_aadhar], required_matches)
                                progress_width = int((progress / required_matches) * 150)
                                cv2.rectangle(im, (x, y + h + 10), (x + progress_width, y + h + 30), (0, 255, 0), -1)
                                cv2.rectangle(im, (x, y + h + 10), (x + 150, y + h + 30), (255, 255, 255), 2)
                                cv2.putText(im, f"{progress}/{required_matches}", (x, y + h + 50), font, 0.6, (255, 255, 255), 1)

                                # Voting Access
                                if match_counts[det_aadhar] >= required_matches:
                                    session['select_aadhar'] = det_aadhar
                                    cam.release()
                                    cv2.destroyAllWindows()
                                    return redirect(url_for('select_candidate'))
                            else:
                                print(f"[WARNING] Aadhaar {det_aadhar} not found in database.")
                                cv2.putText(im, "Unknown", (x, y - 10), font, 0.8, (0, 0, 255), 2)
                        except Exception as e:
                            print(f"[ERROR] Error retrieving Aadhaar: {e}")
                            cv2.putText(im, "Error in retrieveing Person data!", (x, y - 10), font, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(im, "Unknown", (x, y - 10), font, 0.8, (0, 0, 255), 2)

                cv2.imshow('Voting Camera', im)
                try:
                    cv2.setWindowProperty('Voting Camera', cv2.WND_PROP_TOPMOST, 1)
                except:
                    pass

                if cv2.waitKey(1) == ord('q'):
                    break

            flash("Unable to detect person. Contact help desk for manual voting.", "info")
            cam.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"[ERROR] Voting route error: {e}")
            flash("Unexpected error occurred. Please try again.", "danger")
            cam.release()
            cv2.destroyAllWindows()

    return render_template('voting.html')


        

@app.route('/select_candidate', methods=['POST','GET'])
def select_candidate():
    #extract all nominees
    aadhar = session['select_aadhar']

    df_nom=pd.read_sql_query('select * from nominee', mydb)
    all_nom=df_nom['symbol_name'].values
    sq = "select * from vote"
    g = pd.read_sql_query(sq, mydb)
    all_adhar = g['aadhar'].values
    if aadhar in all_adhar:
        flash("You already voted", "warning")
        return redirect(url_for('home'))
    else:
        if request.method == 'POST':
            vote = request.form['test']
            session['vote'] = vote
            sql = "INSERT INTO vote (vote, aadhar) VALUES ('%s', '%s')" % (vote, aadhar)
            cur = mydb.cursor()
            cur.execute(sql)
            mydb.commit()
            cur.close()
            s = "select * from voters where aadhar_id='" + aadhar + "'"
            c = pd.read_sql_query(s, mydb)
            if c.empty:
                flash("Candidate data not found.")
                return redirect(url_for('select_candidate'))  # Replace with appropriate fallback
            pno = str(c.values[0][7])
            name = str(c.values[0][1])
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            url = "https://www.fast2sms.com/dev/bulkV2"

            # message = 'Hi ' + name + ' You voted successfully. Thank you for voting at ' + timeStamp + ' on ' + date + '.'
            no = "9515851969"
            message="helloo hai"
            data1 = {
                "route": "q",
                "message": message,
                "language": "english",
                "flash": 0,
                "numbers": no,
            }

            headers = {
                "authorization": "UwmaiQR5OoA6lSTz93nP0tDxsFEhI7VJrfKkvYjbM2C14Wde8g9lvA2Ghq5VNCjrZ4THWkF1KOwp3Bxd",
                "Content-Type": "application/json"
            }

            response = requests.post(url, headers=headers, json=data1)
            print(response)

            flash(r"Voted Successfully", 'Primary')
            return redirect(url_for('home'))
    return render_template('select_candidate.html', noms=sorted(all_nom))

@app.route('/voting_res')
def voting_res():
    # Fetch vote data
    votes = pd.read_sql_query('SELECT * FROM vote', mydb)

    # Get vote counts and reset index
    counts = votes['vote'].value_counts().reset_index()
    counts.columns = ['symbol', 'count']  # 'symbol' holds symbol image names, 'count' holds vote count

    # Define all possible nominee image filenames (symbol values)
    all_imgs = ['1.png', '2.png', '3.jpg', '4.png', '5.png', '6.png']

    # Prepare frequencies for each symbol safely
    all_freqs = []
    for i in all_imgs:
        match = counts[counts['symbol'] == i]
        if not match.empty:
            all_freqs.append(int(match.iloc[0]['count']))
        else:
            all_freqs.append(0)

    # Fetch nominee names
    df_nom = pd.read_sql_query('SELECT * FROM nominee', mydb)
    all_nom = df_nom['symbol_name'].values

    return render_template('voting_res.html', freq=all_freqs, noms=all_nom)


if __name__=='__main__':
    app.run(debug=True)

