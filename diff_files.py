import difflib
import sys

def read_file_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Remove trailing newlines for comparison but keep empty ones
    return [l.rstrip('\n').rstrip('\r') for l in lines]

def normalize_for_diff(lines):
    """Filter out blank lines to focus on content differences"""
    return [l for l in lines if l.strip()]

def compare_content(zh_path, en_path):
    zh_lines = read_file_lines(zh_path)
    en_lines = read_file_lines(en_path)

    # Filter out empty lines
    zh_content = normalize_for_diff(zh_lines)
    en_content = normalize_for_diff(en_lines)

    print(f"ZH total lines: {len(zh_lines)}")
    print(f"EN total lines: {len(en_lines)}")
    print(f"ZH non-empty lines: {len(zh_content)}")
    print(f"EN non-empty lines: {len(en_content)}")
    print("=" * 80)

    diff = difflib.unified_diff(zh_content, en_content, lineterm='', n=2)
    diff_lines = list(diff)

    with open('diff_content.txt', 'w', encoding='utf-8') as f:
        for line in diff_lines:
            f.write(line + '\n')

    print(f"Content diff written to diff_content.txt ({len(diff_lines)} lines)")

if __name__ == '__main__':
    compare_content(sys.argv[1], sys.argv[2])