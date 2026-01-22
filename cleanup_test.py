with open('weightslab/tests/test_checkpoint_v3_workflow.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with 'if __name__' and keep everything up to and including 'unittest.main(...)'
new_lines = []
in_main = False
main_section_done = False

for i, line in enumerate(lines):
    if "__name__ == '__main__'" in line:
        in_main = True

    new_lines.append(line)

    if in_main and 'unittest.main' in line:
        main_section_done = True
        break

# Remove any lines after main_section_done
if main_section_done:
    with open('weightslab/tests/test_checkpoint_v3_workflow.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f'Cleaned file: kept {len(new_lines)} lines')
