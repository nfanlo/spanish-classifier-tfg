# spanish-classifier-tfg
Trabajo de final de grado sobre la decisión multicriterio

## Git Basics

Download commits from remote repository

```
git pull origin main
```

Upload commits from repository

```
git push origin main
```

Stage changes done in a file
```
git add <filename>
```

Stage changes done in all files
```
git add -A
```

Commit changes
```
git commit -m "<specific message to describe changes>"
```

Check current status of local repo
```
git status
```

Check lineage of commts
```
git log --graph --pretty=oneline --abbrev-commit
```

## **Dev Instructions**

### **Install [Poetry](https://python-poetry.org/docs/)**

e.g.:

`curl -SSL https://install.python-poetry.org | python3 -`

Then, to make avalable the poetry command:

`export PATH="${PATH}":"${HOME}"/.local/bin`

If problems installing Poetry, see [this](https://github.com/python-poetry/install.python-poetry.org/issues/52#issuecomment-1387062081).

Clone repo:

`git clone git@github.com:NFanlo/spanish-classifier-tfg.git`

Move to the repo:

`cd spanish-classifier-tfg`

### **Install deps**

`poetry install`

### **VS Code Integration (Optional)**

`poetry shell  # This activates the virtual environment of the project making it available to VS Code
code .`

Then, in VS Code, choose the interpreter and kernel from the virtual environment.

### **Linting**

Main considerations:

- 120 chars per line (setup your editor for that)
- Google style over pep8 (we can change it if we want to)

If you want to run linting before commit, stage files and then run:

`./project_tasks/lint.sh`

NOTE: If at some point some of the linting errors slows down our goal (e.g. typing with mypy or docstring) we can decide to relax the rules.

