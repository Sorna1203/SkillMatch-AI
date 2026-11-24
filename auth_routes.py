from flask import Blueprint, render_template, request, redirect, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from database.mongo import users_collection
import re

auth_bp = Blueprint("auth", __name__)


def validate_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None


def validate_password(password):
    return len(password) >= 6


@auth_bp.route("/home")
def home():
    """Public landing page."""
    return render_template("index.html")


@auth_bp.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please fill in all fields.", "error")
            return render_template("login.html")

        if not validate_email(email):
            flash("Invalid email format.", "error")
            return render_template("login.html")

        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            session["user"] = email
            return redirect("/dashboard")

        flash("Invalid email or password.", "error")
        return render_template("login.html")

    return render_template("login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please fill in all fields.", "error")
            return render_template("register.html")

        if not validate_email(email):
            flash("Invalid email format.", "error")
            return render_template("register.html")

        if not validate_password(password):
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")

        if users_collection.find_one({"email": email}):
            flash("Email already registered.", "error")
            return render_template("register.html")

        users_collection.insert_one({
            "email":    email,
            "password": generate_password_hash(password)
        })
        flash("Account created! Please sign in.", "success")
        return redirect("/")

    return render_template("register.html")


@auth_bp.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html")


@auth_bp.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


@auth_bp.route("/features")
def features():
    return render_template("features.html")


@auth_bp.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name    = request.form.get("name", "").strip()
        email   = request.form.get("email", "").strip()
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()

        if not all([name, email, subject, message]):
            flash("Please fill in all fields.", "error")
            return render_template("contact.html")

        if not validate_email(email):
            flash("Invalid email format.", "error")
            return render_template("contact.html")

        flash("Thank you! We'll get back to you soon.", "success")
        return render_template("contact.html")

    return render_template("contact.html")