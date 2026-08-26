env_content = """GOOGLE_API_KEY=""
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT="eightfold_ai"
"""

with open('.env', 'w') as f:
    f.write(env_content)

print("✓ .env file created successfully")
print("Add a valid GOOGLE_API_KEY before running Stage 2.")