import git
import os

def get_git_commit_id():
    try:
        # Find the package's directory
        package_path = os.path.dirname(__import__("eval_classification").__file__)
        
        # Initialize the Git repository from the package directory
        repo = git.Repo(package_path, search_parent_directories=True)
        
        # Get the current commit hash
        git_hash = repo.head.object.hexsha
        return git_hash
    except Exception as e:
        print(f"Error retrieving git hash: {e}")
        return "unknown"

# Replace 'your_package_name' with the actual name of the package
__version__ = get_git_commit_id()