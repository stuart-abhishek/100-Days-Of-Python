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