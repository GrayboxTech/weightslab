import subprocess
import os


# --- Git Information Retrieval ---
def get_git_info():
    try:
        # Find git repository root by traversing up from current file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        git_root = current_dir

        # Traverse up to find .git directory
        for _ in range(10): # Limit search depth
            if os.path.isdir(os.path.join(git_root, '.git')):
                break
            parent = os.path.dirname(git_root)
            if parent == git_root: # Reached filesystem root
                git_root = None
                break
            git_root = parent

        if git_root is None:
            return None, None, None

        # Get current git branch
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=git_root, stderr=subprocess.DEVNULL).strip().decode('utf-8')

        # Get current git commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=git_root, stderr=subprocess.DEVNULL).strip().decode('utf-8')

        # Get version (you can modify this if you want a different versioning scheme)
        version = subprocess.check_output(['git', 'describe', '--tags', '--always'], cwd=git_root, stderr=subprocess.DEVNULL).strip().decode('utf-8')

        return branch, version, commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, None, None


# --- Banner Definition ---
branch, version, commit_hash = get_git_info()

# ANSI styling. Color codes are zero-width, so they never affect the art's
# alignment; they're applied per line rather than wrapping the whole block so a
# stray split can't bleed color into the rest of the output.
_RED = "\x1b[31m"
_GREEN = "\x1b[32m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"

_SUBTITLE = "Inspect - Edit - Evolve Neural Networks"
_CREDIT = "By GrayBx"

# "Weightslab" in the FIGlet "big_money-nw" font. Kept as a RAW string so the
# many backslashes in the glyphs stay literal (no invalid-escape warnings, no
# accidental line-continuation). Do NOT reflow or re-indent — the columns must
# stay exactly as generated or the wordmark falls out of alignment.
_WORDMARK = r"""
$$\      $$\           $$\           $$\        $$\               $$\           $$\
$$ | $\  $$ |          \__|          $$ |       $$ |              $$ |          $$ |
$$ |$$$\ $$ | $$$$$$\  $$\  $$$$$$\  $$$$$$$\ $$$$$$\    $$$$$$$\ $$ | $$$$$$\  $$$$$$$\
$$ $$ $$\$$ |$$  __$$\ $$ |$$  __$$\ $$  __$$\\_$$  _|  $$  _____|$$ | \____$$\ $$  __$$\
$$$$  _$$$$ |$$$$$$$$ |$$ |$$ /  $$ |$$ |  $$ | $$ |    \$$$$$$\  $$ | $$$$$$$ |$$ |  $$ |
$$$  / \$$$ |$$   ____|$$ |$$ |  $$ |$$ |  $$ | $$ |$$\  \____$$\ $$ |$$  __$$ |$$ |  $$ |
$$  /   \$$ |\$$$$$$$\ $$ |\$$$$$$$ |$$ |  $$ | \$$$$  |$$$$$$$  |$$ |\$$$$$$$ |$$$$$$$  |
\__/     \__| \_______|\__| \____$$ |\__|  \__|  \____/ \_______/ \__| \_______|\_______/
                           $$\   $$ |
                           \$$$$$$  |
                            \______/
"""

_wordmark_lines = _WORDMARK.strip("\n").split("\n")
_WIDTH = max(len(l) for l in _wordmark_lines)

_banner_lines = [""]
_banner_lines += [f"{_RED}{line}{_RESET}" for line in _wordmark_lines]
_banner_lines += [
    "",
    f"{_GREEN}{_SUBTITLE.center(_WIDTH)}{_RESET}",
    f"{_BOLD}{_CREDIT.center(_WIDTH)}{_RESET}",
    "",
]
_BANNER = "\n".join(_banner_lines) + "\n"

if branch is not None and version is not None and commit_hash is not None:
    _BANNER += f"\nBranch: {branch} | Version: {version} | Commit: {commit_hash}\n"
_BANNER__ = _BANNER # Expose banner with a different name for external use and legacy
