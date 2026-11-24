from flask import Blueprint, render_template, request, session, redirect, flash
from resume.parser import extract_text
from resume.skills import extract_skills
from resume.recommender import (
    recommend_jobs, 
    get_skill_gap, 
    get_motivational_message, 
    get_match_description,
    get_domain_advice, 
    ALL_JOBS
)
from resume.learning_resources import get_learning_resources
from database.mongo import resumes_collection
from datetime import datetime

resume_bp = Blueprint("resume", __name__)

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.png', '.jpg', '.jpeg'}
MAX_FILE_SIZE_MB = 10


@resume_bp.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":
        file = request.files.get("resume")

        if not file or not file.filename:
            flash("Please select a file to upload.", "error")
            return render_template("upload.html")

        # File extension check
        import os
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            flash("Unsupported file type. Please upload PDF, DOCX, PNG, or JPG.", "error")
            return render_template("upload.html")

        # File size check
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            flash(f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.", "error")
            return render_template("upload.html")

        if file_size == 0:
            flash("The uploaded file is empty. Please try a different file.", "error")
            return render_template("upload.html")

        text = extract_text(file)
        if not text or len(text.strip()) < 20:
            flash("Could not extract text from the file. Please try a different file.", "error")
            return render_template("upload.html")

        skills = extract_skills(text)
        if not skills:
            flash("No recognizable skills found in this resume. Try uploading a more detailed resume with specific skill keywords.", "error")
            return render_template("upload.html")

        # Get recommendations with realistic proportional scores
        recommended = recommend_jobs(skills)

        # Filter out jobs with 0% match score - only show relevant jobs
        relevant_jobs = [j for j in recommended if j["score"] > 0]

        # Build job data with realistic scores - show top 15 detailed matches
        top_jobs_data = []
        for job_result in relevant_jobs[:15]:
            gap = get_skill_gap(skills, {"required_skills": job_result["required_skills"]})
            msg = get_motivational_message(job_result["score"], job_result["title"], gap, skills)
            match_desc = get_match_description(job_result["score"])
            top_jobs_data.append({
                "title":    job_result["title"],
                "score":    job_result["score"],
                "domain":   job_result["domain"],
                "gap":      gap,
                "message":  msg,
                "match_description": match_desc,
                "is_primary": job_result["is_primary_domain"],
                "required_skills": job_result["required_skills"]
            })

        # All jobs for full list (filtered to only relevant jobs with score > 0)
        all_jobs_display = [
            {"title": j["title"], "score": j["score"], "domain": j["domain"]}
            for j in relevant_jobs[:50]
        ]

        # Learning resources
        learning_resources = get_learning_resources(skills)

        # Get domain advice (from relevant jobs only)
        primary_domain = relevant_jobs[0]["domain"] if relevant_jobs else "General"
        domain_advice = get_domain_advice(primary_domain, skills)

        # Save to DB
        resumes_collection.insert_one({
            "user":     session["user"],
            "skills":   skills,
            "filename": file.filename,
            "date":     datetime.now(),
            "primary_domain": primary_domain,
            "match_score": recommended[0]["score"] if recommended else 0
        })

        return render_template(
            "recommendation.html",
            skills=skills,
            top_jobs=top_jobs_data,
            all_jobs=all_jobs_display,
            resources=learning_resources,
            domain_advice=domain_advice,
            primary_domain=primary_domain
        )

    return render_template("upload.html")


@resume_bp.route("/history")
def history():
    if "user" not in session:
        return redirect("/")

    user_resumes = list(resumes_collection.find(
        {"user": session["user"]}
    ).sort("date", -1))

    return render_template("history.html", resumes=user_resumes)


@resume_bp.route("/delete/<resume_id>", methods=["POST"])
def delete_resume(resume_id):
    if "user" not in session:
        return redirect("/")

    from bson import ObjectId
    from database.mongo import resumes_collection as rc
    rc.delete_one({"_id": ObjectId(resume_id), "user": session["user"]})
    flash("Resume record deleted successfully.", "success")
    return redirect("/history")


@resume_bp.route("/api/jobs")
def api_jobs():
    """API endpoint to showcase available job roles"""
    return {
        "total_jobs": len(ALL_JOBS),
        "domains": list(set([job["domain"] for job in ALL_JOBS])),
        "jobs": ALL_JOBS
    }
