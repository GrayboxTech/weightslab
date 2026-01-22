import re

with open('weightslab/components/auto_checkpoint.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace escaped triple quotes with regular triple quotes
# Match any backslash followed by triple quotes
content = re.sub(r'\\\\"', '"', content)

with open('weightslab/components/auto_checkpoint.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed auto_checkpoint.py')
