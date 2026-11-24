from flask import Flask
from auth.auth_routes import auth_bp
from resume.resume_routes import resume_bp
from config import SECRET_KEY
from database.mongo import seed_jobs

app = Flask(__name__)
app.secret_key = SECRET_KEY

app.register_blueprint(auth_bp)
app.register_blueprint(resume_bp)

# Seed sample jobs on startup
seed_jobs()

if __name__ == "__main__":
    app.run(debug=True)
