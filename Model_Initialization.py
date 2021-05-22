import git

print("Pulling the model...")
git.Repo.clone_from('https://github.com/johny-kampe/Object-Detection-CV-Lab-2021', 'Object-Detection-CV-Lab-2021')
print("Pull done.")
