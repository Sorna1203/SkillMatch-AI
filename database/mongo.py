from pymongo import MongoClient
from config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

users_collection = db.users
resumes_collection = db.resumes
jobs_collection = db.jobs


def seed_jobs():
    """Seed sample jobs into the database if none exist."""
    if jobs_collection.count_documents({}) == 0:
        sample_jobs = [
            {"title": "Python Developer", "skills": ["python", "sql", "django", "flask", "html", "css"]},
            {"title": "Java Developer", "skills": ["java", "sql", "html", "css", "javascript"]},
            {"title": "Web Developer", "skills": ["html", "css", "javascript", "python", "flask"]},
            {"title": "Data Scientist", "skills": ["python", "machine learning", "data science", "sql", "ai"]},
            {"title": "Full Stack Developer", "skills": ["python", "javascript", "html", "css", "sql", "mongodb"]},
            {"title": "Machine Learning Engineer", "skills": ["python", "machine learning", "ai", "data science", "sql"]},
            {"title": "Frontend Developer", "skills": ["html", "css", "javascript", "python"]},
            {"title": "Backend Developer", "skills": ["python", "sql", "mongodb", "flask", "django"]},
            {"title": "Database Administrator", "skills": ["sql", "mongodb", "python", "c++"]},
            {"title": "AI Developer", "skills": ["ai", "python", "machine learning", "data science"]},
        ]
        jobs_collection.insert_many(sample_jobs)
        print("Sample jobs seeded successfully!")
