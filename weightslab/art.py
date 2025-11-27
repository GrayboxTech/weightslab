import subprocess


# --- Git Information Retrieval ---
def get_git_info():
    try:
        # Get current git branch
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
        
        # Get current git commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        
        # Get version (you can modify this if you want a different versioning scheme)
        version = subprocess.check_output(['git', 'describe', '--tags', '--always']).strip().decode('utf-8')
        
        return branch, version, commit_hash
    except subprocess.CalledProcessError:
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


if __name__ == "__main__":
    print(_BANNER)