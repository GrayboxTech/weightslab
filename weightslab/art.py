import subprocess

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

branch, version, commit_hash = get_git_info()

_BANNER = f"""
 /$$      /$$           /$$           /$$         /$$               /$$                 /$$      
| $$  /$ | $$          |__/          | $$        | $$              | $$                | $$      
| $$ /$$$| $$  /$$$$$$  /$$  /$$$$$$ | $$$$$$$  /$$$$$$    /$$$$$$$| $$        /$$$$$$ | $$$$$$$ 
| $$/$$ $$ $$ /$$__  $$| $$ /$$__  $$| $$__  $$|_  $$_/   /$$_____/| $$       |____  $$| $$__  $$
| $$$$_  $$$$| $$$$$$$$| $$| $$  \ $$| $$  \ $$  | $$    |  $$$$$$ | $$        /$$$$$$$| $$  \ $$
| $$$/ \  $$$| $$_____/| $$| $$  | $$| $$  | $$  | $$ /$$ \____  $$| $$       /$$__  $$| $$  | $$
| $$/   \  $$|  $$$$$$$| $$|  $$$$$$$| $$  | $$  |  $$$$/ /$$$$$$$/| $$$$$$$$|  $$$$$$$| $$$$$$$/
|__/     \__/ \_______/|__/ \____  $$|__/  |__/   \___/  |_______/ |________/ \_______/|_______/ 
                            /$$  \ $$                                                            
                           |  $$$$$$/                                                            
                            \______/                                                             
By GrayBx
Git branch: {branch}
Version: {version}
Commit hash: {commit_hash}
"""
