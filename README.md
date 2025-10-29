

🐍 100 Days of Python Challenge

Welcome to my 100 Days of Python journey!
This repository documents my daily progress in mastering Python from beginner to advanced level.
Each day, I build a new project — starting simple and gradually creating AI-style, data-driven, and logic-based programs.

I’m following this challenge to:

Build a strong foundation in Python 🧠

Learn to think like a programmer 💻

Develop creative and impactful projects 🚀

Prepare myself for IIT Madras → MIT/Stanford Computer Science goals 🎯



---

🧩 Progress Tracker

Day	Project	Description	Status

1	Hello World	My first Python program uploaded to GitHub	✅
2	AI Quote Generator	Interactive motivational quote generator using Python	✅



---

📘 Project Details


---

🧠 Day 1 — Hello World 👋

🔹 Project Title

Hello World Program — My first step into the Python world.

🔹 Project Description

This is my first Python program, created on Day 1 of my 100 Days of Python journey.
It simply prints a welcoming message on the screen and marks the beginning of my coding adventure.

🔹 Code

# Day 1 - Hello World Program
# Author: Stuart Abhishek

print("Hello, World! This is Day 1 of my 100 Days of Python challenge.")

🔹 Example Output

Hello, World! This is Day 1 of my 100 Days of Python challenge.

🔹 What I Learned

How to run my first Python program

How to print messages to the console

Importance of syntax and indentation in Python

The feeling of creating my very first program 💪



---

🧠 Day 2 — AI-Style Quote Generator 🧠

🔹 Project Title

AI Quote Generator — A personalized AI-style motivational quote generator built in Python.

🔹 Project Description

This program asks for your name and your goal, then uses Python’s logic and randomization to generate a unique motivational quote personalized for you.
It uses Python’s random and datetime modules to make the quotes dynamic and time-based.
This project shows creativity, interactivity, and early steps toward AI programming.

🔹 Code

# Day 2 - AI-Style Quote Generator 🧠
# Author: Stuart Abhishek
# Purpose: A small interactive Python program that gives personalized motivational quotes.

import random
import datetime

print("🤖 Welcome to the AI Quote Generator!")
print("Let's create a personalized quote to inspire you today.\n")

# Ask user for details
name = input("What is your name? ")
goal = input("What’s one goal you’re working on right now? ")

# Some smart quotes with placeholders
quotes = [
    f"{name}, remember — every expert was once a beginner. Keep pushing toward {goal}!",
    f"Success doesn’t come from what you do occasionally, {name}, it comes from what you do consistently for {goal}.",
    f"{name}, when you feel like quitting, think about why you started {goal}.",
    f"The future belongs to those like {name} who never stop learning while chasing {goal}.",
    f"{name}, small steps every day towards {goal} will lead to massive results."
]

# Pick a random quote
quote = random.choice(quotes)

# Add a time-based greeting
hour = datetime.datetime.now().hour
if hour < 12:
    greeting = "Good morning"
elif hour < 18:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"

# Final personalized output
print("\n" + "="*60)
print(f"{greeting}, {name}! 🌟")
print("Here’s your motivational message:")
print(f"💬  {quote}")
print("="*60)
print("~ Program created by Stuart Abhishek (Day 2 of 100 Days of Python) ~")

🔹 Example Output

🤖 Welcome to the AI Quote Generator!
Let's create a personalized quote to inspire you today.

What is your name? Stuart
What’s one goal you’re working on right now? Python mastery

============================================================
Good evening, Stuart! 🌟
Here’s your motivational message:
💬  Stuart, when you feel like quitting, think about why you started Python mastery.
============================================================
~ Program created by Stuart Abhishek (Day 2 of 100 Days of Python) ~

🔹 Concepts Used

Variables & Input

Lists and Random Module

String Formatting (f-strings)

Conditional Statements (if-elif-else)

Datetime Module

Code Structuring & Comments


🔹 What I Learned

How to make my code interactive

How to use Python’s built-in modules (random, datetime)

How to make output feel human and intelligent

How to structure readable, professional-looking programs


🔹 Future Improvements

Add more quote categories (study, coding, motivation)

Add color output using colorama

Turn this into a GUI or web-based app in future days
