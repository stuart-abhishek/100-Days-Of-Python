 ---

🐍 100 Days of Python — by Stuart Abhishek

> “Small consistent steps create giant success.” — Stuart Abhishek




---

🌎 About This Repository

Welcome to my 100 Days of Python challenge!

This repository documents my 100-day journey to master Python — one project every day, combining creativity, logic, and real-world problem-solving.
Each project is designed with clean structure, interactive design, and modern programming concepts that reflect the skills of a future Computer Science Engineer at → MIT → Stanford.


---

🎯 My Vision

🔥 Master Python — from fundamentals to advanced algorithms.

💡 Think like an engineer, design like an artist, and build like a scientist.

🚀 Develop projects that prove consistency, logic, and innovation.

🎓 Achieve admission into top universities — MIT, Stanford — through skills and passion.



---

🧩 Progress Tracker

Day	Project	Description	Status

1	Hello World	My first Python program — start of my journey	✅
2	AI Quote Generator	Personalized motivational quote generator	✅
3	Smart Math Quiz	Adaptive arithmetic quiz with scoring & logic	✅
4	Secure Password Engineer	Cryptographically secure password generator + analyzer	✅



---

📘 Project Details


---

🧠 Day 1 — Hello World 👋

🔹 Project Title

Hello World Program — My first step into the world of programming.

🔹 Description

A simple Python script that prints a motivational message.
This marks the beginning of my 100-day journey — the foundation of everything that follows.

🔹 Code

# Day 1 – Hello World Program
# Author: Stuart Abhishek

print("Hello, World! This is Day 1 of my 100 Days of Python challenge.")

🔹 Example Output

Hello, World! This is Day 1 of my 100 Days of Python challenge.

🔹 What I Learned

Basic Python syntax

Printing output to the console

My first taste of programming discipline 💪



---

🧠 Day 2 — AI-Style Quote Generator 🤖

🔹 Project Title

AI Quote Generator — A personalized AI-like motivational quote system.

🔹 Description

This Python script interacts with the user to create motivational quotes based on their name and goal.
It uses randomness and time-based logic to create human-like responses — blending creativity with computation.

🔹 Code

# Day 2 – AI-Style Quote Generator
# Author: Stuart Abhishek

import random, datetime

print("🤖 Welcome to the AI Quote Generator!")
print("Let's create a personalized quote to inspire you today.\n")

name = input("What is your name? ")
goal = input("What’s one goal you’re working on right now? ")

quotes = [
  f"{name}, remember — every expert was once a beginner. Keep pushing toward {goal}!",
  f"Success doesn’t come from what you do occasionally, {name}, it comes from what you do consistently for {goal}.",
  f"{name}, when you feel like quitting, think about why you started {goal}.",
  f"The future belongs to those like {name} who never stop learning while chasing {goal}.",
  f"{name}, small steps every day towards {goal} will lead to massive results."
]

quote = random.choice(quotes)
hour = datetime.datetime.now().hour
greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

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

Randomization

datetime module

String formatting

Conditional statements


🔹 What I Learned

How to make interactive programs

Personalization through logic

Creating “human-feeling” code 🤖



---

🧠 Day 3 — Smart Math Quiz 🎯

🔹 Project Title

Smart Math Quiz — Adaptive arithmetic quiz with scoring & difficulty levels.

🔹 Description

This quiz challenges users with math problems that automatically increase in difficulty as you score higher.
It rewards accuracy, penalizes errors, and gives a professional performance summary — simulating a smart learning system.

🔹 Code

# Day 3 – Smart Math Quiz with Scoring System
# Author: Stuart Abhishek

import random, time

def generate_question(level):
  if level == 1:
    a, b = random.randint(1, 10), random.randint(1, 10)
    op = random.choice(['+', '-'])
  elif level == 2:
    a, b = random.randint(10, 50), random.randint(1, 20)
    op = random.choice(['+', '-', '*'])
  else:
    a, b = random.randint(20, 100), random.randint(1, 25)
    op = random.choice(['+', '-', '*', '//'])
  question = f"{a} {op} {b}"
  return question, eval(question)

def math_quiz():
  print("🧮 Welcome to the Smart Math Quiz!")
  print("Answer as many questions as you can. Type 'quit' to stop.\n")

  level = 1; score = 0; count = 0; start = time.time()
  while True:
    count += 1
    q, ans = generate_question(level)
    user = input(f"Q{count}: {q} = ")
    if user.lower() == "quit": break
    try:
      if int(user) == ans:
        score += 10
        print("✅ Correct!")
        if score % 50 == 0:
          level = min(level + 1, 3)
          print("🚀 Level Up! Difficulty increased.")
      else:
        score -= 5
        print(f"❌ Wrong! Correct answer was {ans}.")
    except ValueError:
      print("⚠️ Enter a number or 'quit'.")
    print(f"Current Score: {score}\n")

  t = round(time.time() - start, 2)
  print("="*55)
  print("🏁 Quiz Summary")
  print(f"Questions: {count - 1} | Final Score: {score} | Time: {t}s")
  if score >= 100: print("🌟 Brilliant work!")
  elif score >= 50: print("💪 Great job!")
  else: print("📘 Keep practicing!")
  print("="*55)
  print("~ Program created by Stuart Abhishek (Day 3 of 100 Days of Python) ~")

if __name__ == "__main__":
  math_quiz()

🔹 Example Output

Q1: 3 + 4 = 7
✅ Correct!
Current Score: 10

Q2: 12 - 8 = 4
✅ Correct!
Current Score: 20

Q3: 6 * 5 = 31
❌ Wrong! Correct answer was 30.
Current Score: 15

🔹 Concepts Used

Functions & Modular Design

Loops & Conditionals

Randomization & Adaptive Logic

Scoring Systems

Time Measurement


🔹 What I Learned

Structured function design

Adaptive algorithms

Logic building like a real engineer ⚙️



---

🧠 Day 4 — Secure Password Engineer 🔐

🔹 Project Title

Secure Password Engineer — Cryptographically secure password generator + strength analyzer.

🔹 Description

This professional-grade program creates uncrackable passwords and analyzes their strength using:

Entropy calculations

Pattern detection

Sequential run identification

Ambiguity checks

Comprehensive 0–100 scoring system


It uses the secrets module (cryptographically secure RNG) and outputs suggestions for improvement — just like a mini cybersecurity assistant.

🔹 Code Highlights

secrets module for secure randomness

math.log2() for entropy estimation

Regular expressions for pattern detection

Logging system using pathlib

Modular functions and clean CLI


🔹 Example Interaction

🔐 Secure Password Engineer — Day 4
1) Generate a strong password
2) Analyze an existing password
3) Generate & analyze (recommended)
q) Quit
Choose an option: 3
Desired length (recommend 16–24): 18
Include lowercase? [Y/n]:
Include uppercase? [Y/n]:
Include digits? [Y/n]:
Include symbols? [Y/n]:
Avoid ambiguous characters? [Y/n]:

Generated password:
7uG}xVbR%pZt_fH3q*

Score: 93/100 | Grade: Very Strong
Length: 18 | Entropy: 113.61 bits
Longest sequential run: 1
Repeated runs: False | Common pattern: False

Suggestions:
• Great length — keep 16+ for stronger security.
• Excellent character diversity — 4 categories used.

🔹 Concepts Used

Cryptography & Secure Randomness

Entropy and Information Theory

Pattern Recognition

Data Validation

File Logging and Modular Programming


🔹 What I Learned

Difference between random and secrets

How to measure password entropy mathematically

Writing security-conscious, user-friendly code

Designing an AI-style analytical program



---

🌟 The Journey Ahead

Day 5 → Mini Calculator App 🧮

Day 6 → AI-Based Password Strength Predictor 🤖

Day 7+ → Data Science, Machine Learning & Real-World Projects


Each project will advance in difficulty and creativity — reflecting both engineering skill and problem-solving ability that top universities like MIT, Stanford, and Harvard value deeply.


---

✍️ Author

👨‍💻 Stuart Abhishek
16-year-old aspiring Computer Science Engineer
Dream Path:→ MIT / Stanford

> “Discipline beats talent. Consistency builds brilliance.”




---

🪪 License

This repository is licensed under the MIT License — free to learn, share, and contribute.


---

💫 Closing Note

> Every single day of this challenge represents my commitment to become a world-class programmer.
Through logic, creativity, and consistent hard work — I’ll reach the top, one line of code at a time.




---