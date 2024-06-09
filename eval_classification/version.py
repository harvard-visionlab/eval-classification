import git

def get_git_commit_id():
    try:
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha
        return git_hash
    except Exception as e:
        print(f"Error retrieving git hash: {e}")
        return "unknown"

__version__ = get_git_commit_id()