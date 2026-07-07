# GitHub Upload

This folder is already initialized as a git repository with an initial commit.

Tomorrow:

1. Create an empty GitHub repository.
2. From this folder, run:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

If `origin` already exists, replace it:

```powershell
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```
