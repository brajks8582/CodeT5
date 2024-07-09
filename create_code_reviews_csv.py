import pandas as pd

# Sample data
data = {
    'code': [
        "def hello_world():\n    print('Hello, world!')",
        "def add(a, b):\n    return a + b",
        "for i in range(5):\n    print(i)"
    ],
    'review': [
        "Consider adding a docstring to describe the function's purpose.",
        "Consider adding type hints for the function parameters.",
        "Consider adding comments to explain the loop."
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('code_reviews.csv', index=False)
