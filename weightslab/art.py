import subprocess
import os


# --- Git Information Retrieval ---
def get_git_info():
    try:
        # Find git repository root by traversing up from current file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        git_root = current_dir

        # Traverse up to find .git directory
        for _ in range(10):  # Limit search depth
            if os.path.isdir(os.path.join(git_root, '.git')):
                break
            parent = os.path.dirname(git_root)
            if parent == git_root:  # Reached filesystem root
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

_BANNER = f"""
\x1b[31m /WW      /WW\x1b[0m           /$$           /$$         /$$               \x1b[32m/$$\x1b[0m                 /$$
\x1b[31m| WW  /W | WW\x1b[0m          |__/          | $$        | $$              \x1b[32m| $$\x1b[0m                | $$
\x1b[31m| WW /WWW| WW\x1b[0m  /$$$$$$  /$$  /$$$$$$ | $$$$$$$  /$$$$$$    /$$$$$$$\x1b[32m| $$\x1b[0m        /$$$$$$ | $$$$$$$
\x1b[31m| WW/WW WW WW\x1b[0m /$$__  $$| $$ /$$__  $$| $$__  $$|_  $$_/   /$$_____/\x1b[32m| $$\x1b[0m       |____  $$| $$__  $$
\x1b[31m| WWWW_  WWWW\x1b[0m| $$$$$$$$| $$| $$  \ $$| $$  \ $$  | $$    |  $$$$$$ \x1b[32m| $$\x1b[0m        /$$$$$$$| $$  \ $$
\x1b[31m| WWW/ \  WWW\x1b[0m| $$_____/| $$| $$  | $$| $$  | $$  | $$ /$$ \____  $$\x1b[32m| $$\x1b[0m       /$$__  $$| $$  | $$
\x1b[31m| WW/   \  WW\x1b[0m|  $$$$$$$| $$|  $$$$$$$| $$  | $$  |  $$$$/ /$$$$$$$/\x1b[32m| $$$$$$$$\x1b[0m  $$$$$$$| $$$$$$$/
\x1b[31m|__/     \__/\x1b[0m \_______/|__/ \____  $$|__/  |__/   \___/  |_______/ \x1b[32m|________/\x1b[0m \_______/|_______/
                            /$$  \ $$
                           |  $$$$$$/
                            \______/
By GrayBx
Git branch: {branch}
Version: {version}
Commit hash: {commit_hash}
"""
