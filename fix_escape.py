import re

# Read the file as binary to see exact bytes
with open('weightslab/components/auto_checkpoint.py', 'rb') as f:
    content = f.read()

# Convert to string
text = content.decode('utf-8', errors='replace')

# Replace all patterns of backslash-escaped quotes with normal quotes
# This handles various escape patterns
text = text.replace('\\"""', '"""')
text = re.sub(r'\\+"""', '"""', text)  # Multiple backslashes
text = re.sub(r'\\\\"', '"', text)  # Escaped quotes

with open('weightslab/components/auto_checkpoint.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed")
