Student Portal System
A comprehensive web-based Student Portal system designed to streamline academic and administrative operations for educational institutions. This robust platform provides distinct dashboards and functionalities for Administrators, Faculty (Teachers), and Students, facilitating efficient data management, communication, and academic tracking. Built with Django (Python) for the backend, HTML, CSS, and JavaScript for a responsive frontend, and MySQL as the primary database, this system aims to simplify educational workflows.

Key Features
The system is meticulously designed to cater to the unique needs of each user role:

1. Admin Module
The administrative interface offers overarching control and management capabilities:

User Management:

Add Faculty: Easily register new teachers, capturing essential details like name, gender, contact information, email, qualifications, teaching experience, and assigned subjects. Each faculty member receives a unique username and password for secure access.

Add Students: Seamlessly enroll new students, recording their personal details, contact information, email, assigned course, and academic year. Students are also provided with secure login credentials.

Data Oversight:

View Faculty: Access a comprehensive list of all registered faculty members and their details.

View Students: Browse and manage records of all enrolled students, including their personal and academic information.

Performance Monitoring:

Performance: (As indicated by AdminScreen.html and requirements.txt which includes numpy, pandas, seaborn, matplotlib) This module is designed to provide analytical insights and visual representations (graphs/charts) of overall school performance, student academic trends, or faculty engagement.

Secure Access:

Login: Dedicated secure login screen (AdminLogin.html) for administrators.

Logout: Securely terminate the administrative session.

2. Faculty Module
Empowering teachers with tools to manage their classes, assignments, and student interactions effectively:

Class Management:

Add Attendance: Record and manage daily or periodic student attendance for their respective courses (AddAttendance.html).

Add Marks: Input and update student marks for assignments, quizzes, and exams (AddMarks.html), along with providing detailed feedback.

Content & Assignment Management:

Create Assignment: Publish new assignments for specific courses, subjects, and student years. Includes fields for assignment task, detailed description, and due date (CreateAssignments.html).

Upload Material: Share educational resources, lecture notes, presentations, and other study materials with their students.

Communication & Tracking:

Messaging: Send direct messages to students.

View Progress Report: Monitor and track the academic progress of individual students or entire classes.

View Attendance: Review historical attendance records for their assigned courses.

Secure Access:

Logout: Securely exit the faculty session.

3. Student Module
Providing students with a personalized and intuitive platform to manage their academic activities and stay informed:

Academic Tracking:

View Upcoming Assignments: Access a centralized view of all pending assignments with their descriptions and deadlines.

View Marks: Check their scores, grades, and feedback for all submitted assignments and exams.

View Upcoming Assignments (Duplicate feature listed by user, but good to emphasize tracking)

Resource Access:

Download Material: Easily download study materials, lecture notes, and other resources uploaded by their faculty members.

Communication:

Message to Faculty: Send messages or queries directly to their teachers for clarifications or support.

View Messages: Review all incoming messages from faculty and administrators.

Secure Access:

Logout: Securely exit the student session.

How to Set Up and Run the Project
To get a local copy of this project running on your system, follow these steps:

Clone the repository:

Bash

git clone https://github.com/NentaShirisha/Student-Portal.git
Navigate into the project directory:

Bash

cd Student-Portal
Create and activate a Python virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate   # On Windows: .\venv\Scripts\activate
Install the required Python packages:

Bash

pip install -r requirements.txt
Run the Django development server:

Bash

python manage.py runserver

├── manage.py          
├── requirements.txt     
├── database.txt        
├── Student/            
│   ├── settings.py      
│   ├── urls.py         
│   └── ...              
├── myapp/              
│   ├── models.py        
│   ├── views.py        
│   ├── urls.py          
│   ├── templates/       
│   │   ├── AdminLogin.html
│   │   ├── AdminScreen.html
│   │   ├── AddFaculty.html
│   │   ├── AddStudent.html
│   │   ├── AddAttendance.html
│   │   ├── AddMarks.html
│   │   ├── CreateAssignments.html
│   │   └── ... (and other view/edit/list pages for faculty and student modules)
│   └── static/         
├── static/              
│   ├── css/
│   ├── js/
│   │   └── datetimepicker.js # As seen in some HTML files for date inputs
│   └── images/
└── media/  
