with open('weightslab/components/auto_checkpoint.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix each line with the escaped quote pattern
fixed_lines = []
for line in lines:
    # Replace backslash-escaped triple quotes with normal triple quotes
    if '\\"\\"\\"\n' in line:
        line = line.replace('\\"\\"\\"\n', '"""\n')
    elif '\\"\\"\\""' in line:
        line = line.replace('\\"\\"\\""', '"""')
    fixed_lines.append(line)

with open('weightslab/components/auto_checkpoint.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print('Fixed auto_checkpoint.py - all escaped triple quotes replaced')
