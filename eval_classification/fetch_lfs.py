import os
import subprocess

def fetch_lfs_files_from_private_repo(repo_url, file_paths, output_root_dir, branch_name='main', tmp_root='~/.tmp'):
    """
        Utility for fetching files from a repo with git-lfs-tracked large files.
        
        Example use:
        repo_url = "git@github.com:harvard-visionlab/configural-shape-analysis.git"
        file_paths = [
            "configural_shape_analysis/results/dnn-evals/eval-classification/imagenet1k_val_2d0d90c3ab/647159ec62_probes.0_resize256_crop224_bilinear_summary.csv",
            "configural_shape_analysis/results/dnn-evals/eval-classification/imagenet1k_val_2d0d90c3ab/647159ec62_probes.1_resize256_crop224_bilinear_summary.csv"
        ]
        output_root_dir = 'data'
        branch_name = 'main'
        tmp_root = '~/.tmp'

        fetch_lfs_files_from_private_repo(repo_url, file_paths, output_root_dir, branch_name, tmp_root)
    """
    # Expand the user directory if ~ is used
    tmp_root = os.path.expanduser(tmp_root)
    
    # Clone the repository into a temporary directory within tmp_root
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    temp_dir = os.path.join(tmp_root, f'temp_{repo_name}')
    if os.path.exists(temp_dir):
        subprocess.run(['rm', '-rf', temp_dir])
    
    os.makedirs(tmp_root, exist_ok=True)
    
    # Clone the repository with sparse-checkout enabled
    subprocess.run(['git', 'clone', '--branch', branch_name, '--depth', '1', '--no-checkout', repo_url, temp_dir])
    
    # Set sparse-checkout to only include the specified files
    subprocess.run(['git', '-C', temp_dir, 'config', 'core.sparseCheckout', 'true'])
    
    sparse_checkout_file = os.path.join(temp_dir, '.git', 'info', 'sparse-checkout')
    os.makedirs(os.path.dirname(sparse_checkout_file), exist_ok=True)
    
    with open(sparse_checkout_file, 'w') as f:
        for file_path in file_paths:
            f.write(file_path + '\n')
    
    # Fetch the specified files from the remote repository
    subprocess.run(['git', '-C', temp_dir, 'checkout', branch_name])
    
    # Sanity check to ensure only the specified files are downloaded, ignoring the .git folder
    downloaded_files = []
    for root, dirs, files in os.walk(temp_dir):
        if '.git' in root:
            continue
        for file in files:
            downloaded_files.append(os.path.relpath(os.path.join(root, file), temp_dir))
    
    print("Downloaded files:")
    for df in downloaded_files:
        print(df)
    
    expected_files_set = set(file_paths)
    downloaded_files_set = set(downloaded_files)
    if downloaded_files_set == expected_files_set:
        print("Sanity check passed: Only the specified files are downloaded.")
    else:
        print("Sanity check failed: Additional files were downloaded.")
    
    # Ensure the directory structure exists in the output_root_dir and move the files
    for file_path in file_paths:
        source_file = os.path.join(temp_dir, file_path)
        if os.path.exists(source_file):
            dest_dir = os.path.join(output_root_dir, os.path.dirname(file_path))
            os.makedirs(dest_dir, exist_ok=True)
            dest_file = os.path.join(output_root_dir, file_path)
            subprocess.run(['mv', source_file, dest_file])
        else:
            print(f"File {file_path} not found in the repository.")

    # Clean up the temporary directory
    subprocess.run(['rm', '-rf', temp_dir])