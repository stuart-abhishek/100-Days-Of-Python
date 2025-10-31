# Day 5 — Natural-Language Smart Calculator 🤖
# Author: Stuart Abhishek
# Description: Understands text expressions like "add 24 and 65"
# and computes the result intelligently.

import re
import math

def extract_numbers(text):
    """Return all numbers in the sentence as floats."""
    return [float(n) for n in re.findall(r"-?\d+\.?\d*", text)]

def calculate_from_text(text):
    """Interpret natural-language math commands and compute the answer."""
    text = text.lower().strip()
    nums = extract_numbers(text)
    if not nums:
        return "❌ No numbers found."

    # Addition
    if any(w in text for w in ["add", "plus", "sum"]):
        return sum(nums)

    # Subtraction
    if "subtract" in text or "minus" in text:
        if "from" in text and len(nums) == 2:
            return nums[1] - nums[0]
        return nums[0] - sum(nums[1:])

    # Multiplication
    if any(w in text for w in ["multiply", "times", "product", "×", "x"]):
        result = 1
        for n in nums:
            result *= n
        return result

    # Division
    if any(w in text for w in ["divide", "divided", "over", "÷", "by"]):
        try:
            result = nums[0]
            for n in nums[1:]:
                result /= n
            return result
        except ZeroDivisionError:
            return "⚠️ Division by zero not allowed."

    # Power / Square / Cube
    if "square" in text and len(nums) == 1:
        return nums[0] ** 2
    if "cube" in text and len(nums) == 1:
        return nums[0] ** 3
    if "power" in text or "^" in text:
        return math.pow(nums[0], nums[1])

    # Square root
    if "square root" in text or "√" in text:
        return math.sqrt(nums[0])

    # Percentage
    if "%" in text or "percent" in text:
        if "of" in text and len(nums) == 2:
            return (nums[0] / 100) * nums[1]
        return nums[0] / 100

    return "🤔 Sorry, I couldn’t understand that operation."

def main():
    print("🧮 Welcome to the Natural-Language Smart Calculator!")
    print("Type a math sentence (e.g., 'add 45 and 62', 'square root of 81')")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("👉 Enter expression: ")
        if query.lower().strip() in {"quit", "exit"}:
            print("Goodbye! Keep learning mathematics and logic. 🚀")
            break

        result = calculate_from_text(query)
        print(f"✅ Result: {result}\n")

if __name__ == "__main__":
    main()