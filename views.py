from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pickle
import pymysql
import os
from django.core.files.storage import FileSystemStorage
from datetime import datetime
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import smtplib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


global uname, graph
dataset = pd.read_csv("model/dataset.csv")
labels = np.unique(dataset['FinalResult'])
le = LabelEncoder()
dataset['FinalResult'] = pd.Series(le.fit_transform(dataset['FinalResult'].astype(str)))#encode all str columns to numeric

Y = dataset['FinalResult'].ravel()
dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
#print(X)
sc = MinMaxScaler()
X = sc.fit_transform(X)
#print(X)
#print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

accuracy = []
precision = []
recall = [] 
fscore = []

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    global graph
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    '''
    conf_matrix = confusion_matrix(y_test, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.title("Teacher's Effectiveness Based on Student Performance Grades")
    plt.close()
    graph = base64.b64encode(buf.getvalue()).decode()
    '''
    
nb_cls = GaussianNB()
nb_cls.fit(X_train, y_train)
predict = nb_cls.predict(X_test)
calculateMetrics("Naive Bayes", y_test, predict)

xg_cls = XGBClassifier()
xg_cls.fit(X_train, y_train)
predict = xg_cls.predict(X_test)
calculateMetrics("XGBoost", y_test, predict)

def runML(marks):
    global xg_cls, sc, labels
    data = []
    data.append([marks])
    data = np.asarray(data)
    data = sc.transform(data)
    predict = xg_cls.predict(data)[0]
    return labels[predict]

def ViewMarks(request):
    if request.method == 'GET':
       return render(request, 'ViewMarks.html', {}) 

def ViewMarksAction(request):
    if request.method == 'POST':
        u = request.session.get('student_name', '').strip()
        c = request.POST.get('t1', '').strip()
        y = request.POST.get('t2', '').strip()

        print("DEBUG VALUES:")
        print(f"Student name from session: {u!r}")
        print(f"Course from form: {c!r}")
        print(f"Year from form: {y!r}")

       # if not all([uname, course, year]):
        #    return render(request, 'error.html', {'message': 'Missing required input values. Please log in and try again.'})

        total = 0
        count = 0
        output = '<table border=1 align=center width=100%><tr><th><font size="" color="black">Subject Name</th><th><font size="" color="black">Obtained Marks</th></tr>'

        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='studentportal', charset='utf8')
        with con:
            cur = con.cursor()
            #cur.execute("SELECT subject_name, subject_marks FROM marks WHERE student_name='Student' AND course_year='I' AND course_name='python'")
            cur.execute("SELECT subject_name, subject_marks FROM marks WHERE student_name=%s AND course_year=%s AND course_name=%s", (u, y, c))
            rows = cur.fetchall()
            #print(f"DEBUG: rows fetched = {rows}")
            print(f"DEBUG: Rows fetched = {len(rows)}")
            for row in rows:
                print(f"DEBUG ROW: {row}")
                output += '<tr><td><font size="" color="black">' + row[0] + '</td><td><font size="" color="black">' + str(row[1]) + '</td></tr>'
                total += row[1]
                count += 1

        if count == 0:
            avg = 0
        else:
            avg = total / count

        output += '<tr><td>-</td><td>-</td></tr>'
        output += '<tr><td><font size="" color="black">Total Marks</td><td><font size="" color="black">' + str(total) + '</td></tr>'
        output += '<tr><td><font size="" color="black">Average GPA</td><td><font size="" color="black">' + str(avg) + '</td></tr>'

        feedback_ml = runML(avg)
        output += '<tr><td><font size="" color="black">ML Predicted Feedback</td><td><font size="" color="black">' + feedback_ml + '</td></tr>'
        output += "</table></br></br></br></br>"

        context = {'data': output}
        return render(request, 'StudentScreen.html', context)
  


def ViewMessages(request):
    if request.method == 'GET':
        uname = request.session.get('student_name')

        # Check if session variable exists
        #if not uname:
         #   return render(request, 'error.html', {'message': 'Student not logged in or session expired.'})

        output = '<table border=1 align=center width=100%><tr><th><font size="" color="black">Sender Name</th><th><font size="" color="black">Subject</th>'
        output += '<th><font size="" color="black">Message</th><th><font size="" color="black">Message Date</th></tr>'

        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='studentportal', charset='utf8')
        with con:
            cur = con.cursor()
            # Use parameterized query to prevent SQL injection and NoneType errors
            cur.execute("SELECT * FROM messages WHERE receiver_name = %s", (uname,))
            rows = cur.fetchall()
            for row in rows:
                output += '<tr><td><font size="" color="black">' + str(row[0]) + '</td><td><font size="" color="black">' + str(row[2]) + '</td>'
                output += '<td><font size="" color="black">' + str(row[3]) + '</td><td><font size="" color="black">' + str(row[4]) + '</td></tr>'

        output += "</table></br></br></br></br>"
        context = {'data': output}
        return render(request, 'StudentScreen.html', context)


def DownloadMaterialAction(request):
    if request.method == 'GET':
        filename = request.GET.get('name', False)
        with open("StudentApp/static/files/"+filename, "rb") as file:
            content = file.read()
        file.close()
        response = HttpResponse(content,content_type='application/force-download')
        response['Content-Disposition'] = 'attachment; filename='+filename
        return response

def DownloadMaterials(request):
    if request.method == 'GET':
        
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Faculty Name</th><th><font size="" color="black">Material Name</th>'
        output+='<th><font size="" color="black">Description</th><th><font size="" color="black">Filename</th>'
        output+='<th><font size="" color="black">Upload Date</th><th><font size="" color="black">Click Here to Download</th>'
        output+='</tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from uploadmaterial")
            rows = cur.fetchall()
            for row in rows:
               stored_filename = str(row[3] or '')
               if '_' in stored_filename:
                   display_filename = stored_filename.split('_', 1)[1]  # part after first underscore
               else:
                   display_filename = stored_filename
               print(f"DEBUG: stored='{stored_filename}', display='{display_filename}'")
               output += '<tr>'
               output += '<td><font size="" color="black">' + str(row[0] or '') + '</td>'
               output += '<td><font size="" color="black">' + str(row[1] or '') + '</td>'
               output += '<td><font size="" color="black">' + str(row[2] or '') + '</td>'
               output += '<td><font size="" color="black">' + display_filename + '</td>'
               output += '<td><font size="" color="black">' + str(row[4] or '') + '</td>'
               output += '<td><a href=\'DownloadMaterialAction?name=' + stored_filename + '\'><font size=3 color=black>Download</font></a></td>'
               output += '</tr>'

        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'StudentScreen.html', context) 

from datetime import datetime

def StudentMessagingAction(request):
    if request.method == 'POST':
        uname = request.session.get('student_name')
        fname = request.POST.get('sname')
        subject = request.POST.get('t1')
        message = request.POST.get('t2')

        # Check for missing values and handle gracefully
        if not uname or not fname or not subject or not message:
            status = "All fields are required."
            return render(request, 'StudentScreen.html', {'data': status})

        now = datetime.now()
        current_datetime = now.strftime("%Y-%m-%d")  # Date only as string

        db_connection = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            database='studentportal',
            charset='utf8'
        )
        db_cursor = db_connection.cursor()

        # Use parameterized query to avoid SQL injection and handle None properly
        student_sql_query = "INSERT INTO messages (sender_name, receiver_name, subject, message, message_date) VALUES (%s, %s, %s, %s, %s)"
        db_cursor.execute(student_sql_query, (uname, fname, subject, message, current_datetime))

        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")

        if db_cursor.rowcount == 1:
            status = "Message successfully sent to teacher " + fname
        else:
            status = "Failed to send message."

        return render(request, 'StudentScreen.html', {'data': status})


def StudentMessaging(request):
    if request.method == 'GET':
    
        output = '<tr><td><font size="3" color="black"><b>Choose&nbsp;Teachert&nbsp;Name</b></td>'
        output += '<td><select name="sname">'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from faculty")
            rows = cur.fetchall()
            for row in rows:
                output += '<option value="'+row[0]+'">'+row[0]+'</option>'
        output += "</select></td></tr>"
        context= {'data1': output}
        return render(request, 'StudentMessaging.html', context)

def ViewAssignments(request):
    if request.method == 'GET':

        def safe_str(val):
            return '' if val is None else str(val)

        output = '<table border=1 align=center width=100%><tr><th><font size="" color="black">Faculty Name</th><th><font size="" color="black">Course Name</th>'
        output += '<th><font size="" color="black">Subject Name</th><th><font size="" color="black">Course Year</th>'
        output += '<th><font size="" color="black">Assignment Task</th><th><font size="" color="black">Description</th>'
        output += '<th><font size="" color="black">Assignment Date</th></tr>'

        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='studentportal', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from assignments")
            rows = cur.fetchall()
            for row in rows:
                output += '<tr>'
                output += '<td><font size="" color="black">' + safe_str(row[0]) + '</td>'
                output += '<td><font size="" color="black">' + safe_str(row[1]) + '</td>'
                output += '<td><font size="" color="black">' + safe_str(row[2]) + '</td>'
                output += '<td><font size="" color="black">' + safe_str(row[3]) + '</td>'
                output += '<td><font size="" color="black">' + safe_str(row[4]) + '</td>'
                output += '<td><font size="" color="black">' + safe_str(row[5]) + '</td>'
                output += '<td><font size="" color="black">' + safe_str(row[6]) + '</td>'
                output += '</tr>'

        output += "</table></br></br></br></br>"
        context = {'data': output}
        return render(request, 'StudentScreen.html', context)


def ViewStudentMessages(request):
    if request.method == 'GET':
        uname = request.session.get('faculty_name')  
        
        if not uname:
            # Handle missing username gracefully
            return render(request, 'FacultyScreen.html', {'data': 'User not logged in or username not found.'})
        
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Sender Name</th><th><font size="" color="black">Subject</th>'
        output+='<th><font size="" color="black">Message</th><th><font size="" color="black">Message Date</th>'
        output+='</tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from messages where receiver_name='"+uname+"'")
            rows = cur.fetchall()
            for row in rows:
                output+='<tr><td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[2])+'</td>'
                output+='<td><font size="" color="black">'+row[3]+'</td><td><font size="" color="black">'+str(row[4])+'</td></tr>'                
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'FacultyScreen.html', context)

def ViewProgressReportAction(request):
    if request.method == 'POST':
        
        sname = request.POST.get('sname', False)
        course = request.POST.get('t1', False)
        year = request.POST.get('t2', False)
        total = 0
        count = 0
        output = '<table border=1 align=center width=100%><tr><th><font size="" color="black">Subject Name</th><th><font size="" color="black">Obtained Marks</th></tr>'
        
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='studentportal', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select subject_name,subject_marks from marks where student_name=%s and course_year=%s and course_name=%s", (sname, year, course))
            rows = cur.fetchall()
            for row in rows:
                output += '<tr><td><font size="" color="black">' + row[0] + '</td><td><font size="" color="black">' + str(row[1]) + '</td></tr>'
                total += row[1]
                count += 1
        
        average = total / count if count != 0 else 0
        
        output += '<tr><td>-</td><td>-</td></tr>'
        output += '<tr><td><font size="" color="black">Total Marks</td><td><font size="" color="black">' + str(total) + '</td></tr>'
        output += f'<tr><td><font size="" color="black">Average GPA</td><td><font size="" color="black">{average}</td></tr>'
        
        feedback_ml = runML(average)
        output += f'<tr><td><font size="" color="black">ML Predicted Feedback</td><td><font size="" color="black">{feedback_ml}</td></tr>'
        output += "</table></br></br></br></br>"
        
        context = {'data': output}
        return render(request, 'FacultyScreen.html', context)
    

def ViewProgressReport(request):
    if request.method == 'GET':
        output = '<tr><td><font size="3" color="black"><b>Choose&nbsp;Student&nbsp;Names</b></td>'
        output += '<td><select name="sname">'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from student")
            rows = cur.fetchall()
            for row in rows:
                output += '<option value="'+row[0]+'">'+row[0]+'</option>'
        output += "</select></td></tr>"
        context= {'data1': output}
        return render(request, 'ViewProgressReport.html', context)

def sendMail(subject, msg, email):
    print("sending reminder to mail")
    em = []
    em.append(email)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:
        email_address = 'kaleem202120@gmail.com'
        email_password = 'xyljzncebdxcubjq'
        connection.login(email_address, email_password)
        connection.sendmail(from_addr="kaleem202120@gmail.com", to_addrs=em, msg="Subject : "+subject+"\n"+msg)   

def getEmail(sname):
    email = ""
    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select email from student where username='"+sname+"'")
        rows = cur.fetchall()
        for row in rows:
            email = row[0]
            break
    return email    

def MessagingAction(request):
    if request.method == 'POST':
        try:
            # ✅ Get uname FIRST
            uname = request.POST.get('uname', '').strip()
            sname = request.POST.get('sname', '').strip()
            subject = request.POST.get('t1', '').strip()
            message = request.POST.get('t2', '').strip()

            # ✅ Validate
            if not all([uname, sname, subject, message]):
                return render(request, 'FacultyScreen.html', {'data': 'All fields are required'})

            current_date = datetime.now().strftime('%Y-%m-%d')

            # ✅ DB connection
            db_connection = pymysql.connect(
                host='127.0.0.1',
                port=3306,
                user='root',
                password='root',
                database='studentportal',
                charset='utf8'
            )
            db_cursor = db_connection.cursor()

            # ✅ Insert query using correct column names
            insert_query = """
                INSERT INTO messages (sender_name, receiver_name, subject, message, message_date)
                VALUES (%s, %s, %s, %s, %s)
            """
            db_cursor.execute(insert_query, (uname, sname, subject, message, current_date))
            db_connection.commit()

            status = "Message sent successfully to " + sname if db_cursor.rowcount == 1 else "Message failed to send"

            db_cursor.close()
            db_connection.close()

        except Exception as e:
            status = f"Error: {str(e)}"

        return render(request, 'FacultyScreen.html', {'data': status})

def Messaging(request):
    if request.method == 'GET':
        output = '<tr><td><font size="3" color="black"><b>Choose&nbsp;Student&nbsp;Name</b></td>'
        output += '<td><select name="sname">'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from student")
            rows = cur.fetchall()
            for row in rows:
                output += '<option value="'+row[0]+'">'+row[0]+'</option>'
        output += "</select></td></tr>"
        context= {'data1': output}
        return render(request, 'Messaging.html', context)

def AddMarksAction(request):
    if request.method == 'POST':
        sname = request.POST.get('sname', '').strip()
        uname = request.session.get('faculty_name', '').strip()
        print("DEBUG: faculty_name from session =", uname)  
        status = "Error in adding marks details"
        try:
            uname = request.session.get('faculty_name', '').strip()
            print("DEBUG: faculty_name from session =", uname)  

            sname = request.POST.get('sname', '').strip()
            #sname = request.POST.get('sname', '').strip()
            course = request.POST.get('t1', '').strip()
            subject = request.POST.get('t2', '').strip()
            year = request.POST.get('t3', '').strip()
            marks = request.POST.get('t4', '').strip()
            feedback = request.POST.get('t5', '').strip()

            # Uncomment below if you want to enforce required fields
            # if not all([uname, sname, course, subject, year, marks]):
            #     return render(request, 'FacultyScreen.html', {'data': 'Please fill all required fields.'})

            current_datetime = datetime.now().strftime("%Y-%m-%d")

            db_connection = pymysql.connect(
                host='127.0.0.1',
                port=3306,
                user='root',
                password='root',
                database='studentportal',
                charset='utf8'
            )
            db_cursor = db_connection.cursor()

            insert_query = """
                INSERT INTO marks 
                (student_name, faculty_name, course_name, course_year, subject_name, subject_marks, feedback, upload_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            db_cursor.execute(insert_query, (sname, uname, course, year, subject, marks, feedback, current_datetime))
            db_connection.commit()

            if db_cursor.rowcount == 1:
                status = "Marks details successfully submitted"

            db_cursor.close()
            db_connection.close()

        except Exception as e:
            status = f"Error: {str(e)}"

        return render(request, 'FacultyScreen.html', {'data': status})


def AddMarks(request):
    if request.method == 'GET':
        output = '<tr><td><font size="3" color="black"><b>Choose&nbsp;Student&nbsp;Names</b></td>'
        output += '<td><select name="sname">'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from student")
            rows = cur.fetchall()
            for row in rows:
                output += '<option value="'+row[0]+'">'+row[0]+'</option>'
        output += "</select></td></tr>"
        context= {'data1': output}
        return render(request, 'AddMarks.html', context)

from django.shortcuts import render
from datetime import datetime
import os
import pymysql
import uuid

def UploadMaterialAction(request):
    if request.method == 'POST':
        fname = request.session.get('faculty_name')
        print("Session faculty_name retrieved:", fname)
        status = "Error in uploading material details"
        try:
            # Get form data
            uname = fname
            material = request.POST.get('t1')
            desc = request.POST.get('t2')
            file_obj = request.FILES['t3']
            
            filename = f"{uuid.uuid4().hex}_{file_obj.name}"
            

            # Get current date
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Save file safely using Django's storage
            fs = FileSystemStorage(location="StudentApp/static/files/")
            if fs.exists(filename):
                fs.delete(filename)  # Safely delete existing file
            fs.save(filename, file_obj)  # Save uploaded file

            # Save record to database
            db_connection = pymysql.connect(
                host='127.0.0.1',
                port=3306,
                user='root',
                password='root',
                database='studentportal',
                charset='utf8'
            )
            db_cursor = db_connection.cursor()

            insert_query = """
                INSERT INTO uploadmaterial (faculty_name, material_name, description, filename, upload_date)
                VALUES (%s, %s, %s, %s, %s)
            """
            db_cursor.execute(insert_query, (uname, material, desc, filename, current_date))
            db_connection.commit()

            if db_cursor.rowcount == 1:
                status = "Material details added to database"

            db_cursor.close()
            db_connection.close()

        except Exception as e:
            status = f"Error: {str(e)}"

        return render(request, 'UploadMaterial.html', {'data': status})


def UploadMaterial(request):
    if request.method == 'GET':
       return render(request, 'UploadMaterial.html', {}) 

def CreateAssignmentsAction(request):
    if request.method == 'POST':
        fname = request.session.get('faculty_name')
        print("Session faculty_name retrieved:", fname)
        course = request.POST.get('course_name')
        subject = request.POST.get('subject_name')
        year = request.POST.get('course_year')
        task = request.POST.get('assignment_task')
        desc = request.POST.get('description')
        adate = request.POST.get('assignment_date')

        db = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='root',
            database='studentportal',
            charset='utf8'
        )
        cursor = db.cursor()

        sql = """
            INSERT INTO assignments 
            (faculty_name, course_name, subject_name, course_year, assignment_task, description, assignment_date) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (fname, course, subject, year, task, desc, adate))
        db.commit()

        if cursor.rowcount == 1:
            message = "Assignment created successfully"
        else:
            message = "Failed to create assignment"

        cursor.close()
        db.close()

        return render(request, 'FacultyScreen.html', {'data': message})


def CreateAssignments(request):
    if request.method == 'GET':
       return render(request, 'CreateAssignments.html', {}) 

def AddAttendanceAction(request):
    if request.method == 'POST':
        uname = request.session.get('faculty_name')

       
        students = request.POST.getlist('t1')
        current_date = datetime.now().strftime("%Y-%m-%d")

        db_connection = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='root',
            database='studentportal',
            charset='utf8'
        )
        db_cursor = db_connection.cursor()

        for student in students:
            # Use parameterized query to avoid SQL injection
            sql = "INSERT INTO student_attendance (student_name, faculty_name, attendance_date) VALUES (%s, %s, %s)"
            db_cursor.execute(sql, (student, uname, current_date))

        db_connection.commit()
        db_cursor.close()
        db_connection.close()

        context = {'data': "Selected Students Attendance Marked Successfully"}
        return render(request, 'FacultyScreen.html', context)
  
            

def AddAttendance(request):
    if request.method == 'GET':
        output = '<tr><td><font size="3" color="black"><b>Choose&nbsp;Student&nbsp;Names</b></td>'
        output += '<td><select name="t1" multiple>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from student")
            rows = cur.fetchall()
            for row in rows:
                output += '<option value="'+row[0]+'">'+row[0]+'</option>'
        output += "</select></td></tr>"
        context= {'data1': output}
        return render(request, 'AddAttendance.html', context)   

def SchoolPerformance(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Teacher Name</th><th><font size="" color="black">Average Students Performance Grade</th>'
        output+='</tr>'
        scores = []
        labels = []
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select faculty_name, avg(subject_marks) from marks group by faculty_name")
            rows = cur.fetchall()
            for row in rows:
                scores.append(row[1])
                labels.append(row[0])
                output+='<tr><td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[1])+'</td></tr>'
        output+= "</table></br>"        
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        plt.pie(scores, labels=labels, autopct='%1.1f%%')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.title("Teacher's Effectiveness Based on Student Performance Grades")
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'AdminScreen.html', context)         

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})    

def StudentLogin(request):
    if request.method == 'GET':
       return render(request, 'StudentLogin.html', {})

def AddFaculty(request):
    if request.method == 'GET':
       return render(request, 'AddFaculty.html', {})

def AddStudent(request):
    if request.method == 'GET':
       return render(request, 'AddStudent.html', {})    

def index(request):
    if request.method == 'GET':
        global graph, accuracy, precision, recall, fscore
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output += '<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th>'
        output+='</tr>'
        algorithms = ['Naive Bayes', 'XGBoost']
        for i in range(len(accuracy)):
            output+='<tr><td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td>'
            output+='<td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td>'
            output+='<td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>" 
        context= {'data':output}
        return render(request, 'index.html', context)   

def FacultyLogin(request):
    if request.method == 'GET':
       return render(request, 'FacultyLogin.html', {})

def AdminLoginAction(request):
    if request.method == 'POST':
        
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            context= {'data':'welcome '+username}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'AdminLogin.html', context)

def FacultyLoginAction(request):
    if request.method == 'POST':
        
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if "'" not in username:
            username = "'"+username+"'"
        if "'" not in password:
            password = "'"+password+"'"    
        print(username+" === "+password)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username, password FROM faculty where username="+username+" and password="+password)
            rows = cur.fetchall()
            for row in rows:
                uname = username
                index = 1
                break		
        if index == 1:
            request.session['faculty_name'] = username.strip("'")
            print("Session faculty_name set to:", request.session['faculty_name'])

            context= {'data':'welcome '+username}
            return render(request, 'FacultyScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'FacultyLogin.html', context)        
    
def StudentLoginAction(request):
    if request.method == 'POST':
        
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username, password FROM student")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            request.session['student_name'] = username
            context= {'data':'welcome '+username}
            return render(request, 'StudentScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'StudentLogin.html', context)

def ViewFaculty(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Faculty Name</th><th><font size="" color="black">Gender</th>'
        output+='<th><font size="" color="black">Contact No</th><th><font size="" color="black">Email ID</th>'
        output+='<th><font size="" color="black">Qualification</th><th><font size="" color="black">Experience</th>'
        output+='<th><font size="" color="black">Teaching Subjects</th>'
        output+='<th><font size="" color="black">Username</th><th><font size="" color="black">Password</th></tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from faculty")
            rows = cur.fetchall()
            output+='<tr>'
            for row in rows:
                output+='<td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+row[2]+'</td><td><font size="" color="black">'+row[3]+'</td>'
                output+='<td><font size="" color="black">'+row[4]+'</td><td><font size="" color="black">'+row[5]+'</td>'
                output+='<td><font size="" color="black">'+row[6]+'</td>'
                output+='<td><font size="" color="black">'+row[7]+'</td>'
                output+='<td><font size="" color="black">'+row[8]+'</td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)    

def AddFacultyAction(request):
    if request.method == 'POST':
        faculty = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        qualification = request.POST.get('t5', False)
        experience = request.POST.get('t6', False)
        teaching = request.POST.get('t7', False)
        username = request.POST.get('t8', False)
        password = request.POST.get('t9', False)
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username FROM faculty")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    status = "Username already exists"
                    break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO faculty(faculty_name,gender,contact_no,email,qualification,experience,teaching_subjects,username,password) VALUES('"+faculty+"','"+gender+"','"+contact+"','"+email+"','"+qualification+"','"+experience+"','"+teaching+"','"+username+"','"+password+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "Faculty details added"
        context= {'data': status}
        return render(request, 'AddFaculty.html', context)

def ViewStudent(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Student Name</th><th><font size="" color="black">Gender</th>'
        output+='<th><font size="" color="black">Contact No</th><th><font size="" color="black">Email ID</th>'
        output+='<th><font size="" color="black">Course Name</th><th><font size="" color="black">Year</th>'
        output+='<th><font size="" color="black">Username</th><th><font size="" color="black">Password</th></tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from student")
            rows = cur.fetchall()
            output+='<tr>'
            for row in rows:
                output+='<td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+row[2]+'</td><td><font size="" color="black">'+row[3]+'</td>'
                output+='<td><font size="" color="black">'+row[4]+'</td><td><font size="" color="black">'+row[5]+'</td>'
                output+='<td><font size="" color="black">'+row[6]+'</td>'
                output+='<td><font size="" color="black">'+row[7]+'</td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)  

def AddStudentAction(request):
    if request.method == 'POST':
        student = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        course = request.POST.get('t5', False)
        year = request.POST.get('t6', False)
        username = request.POST.get('t7', False)
        password = request.POST.get('t8', False)
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username FROM student")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    status = "Username already exists"
                    break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'studentportal',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO student(student_name,gender,contact_no,email,course,course_year,username,password) VALUES('"+student+"','"+gender+"','"+contact+"','"+email+"','"+course+"','"+year+"','"+username+"','"+password+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "Student details added"
        context= {'data': status}
        return render(request, 'AddStudent.html', context)
    
