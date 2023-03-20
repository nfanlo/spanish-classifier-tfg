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

### **Install [Pyenv](https://github.com/pyenv/pyenv#homebrew-in-macos)**

Before we start, let's go to our $HOME/dev directory (create it if not present):

```sh
mkdir ${HOME}/dev
cd ${HOME}/dev
```

Now you shoud be in your development directory, e.g. `/Users/fperez/dev` (You can check the current directory where you are with the command `pwd`)

Then, let's update `brew` and install `pyenv`
```sh
brew update
brew install pyenv
```

Then, for the Zsh, execute in the terminal:
```sh
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

Then restart the terminal to allow the previous changes to work!!!

If you executed the previous steps correctly, now you would be able to show all the python available versions at this time with `pyenv` lets execute:

```sh
pyenv versions
```

This will show us something like this:
```
  system
  3.6.0
* anaconda3-4.3.0 (set by /Users/fperez/.python-version)
```

Let's show the local Python version with `pyenv`, which should be the same shown when executing `python --version`
```sh
pyenv local
```
This should show your current local active version of Python. In my case shows the Python version corresponding to the * shown above (anaconda3-4.3.0 which corresponds to Python version 3.6.0)

Same applies for the global version:
```sh
pyenv global
```

Let's install a new specific global version. We will use it later in this project with Poetry:
```sh
pyenv install 3.9.12
```

Now the 3.9.12 version of Python should be installed but not set as a default neither as local nor global Python versions yet. Let's check this:

```sh
pyenv version
```
Should still show us the previous version present in your system (the same one as shown above for the `pyenv local`, 3.6.0 in my case).

Let's set 3.9.12 as the default version:

```sh
pyenv global 3.9.12
pyenv local 3.9.12
```

Doing this, sets the global version in the file `~/.pyenv/version` and the local version in `~/.python-version` both to 3.9.12 (you can check the content of these files with `cat ~/.pyenv/version` and `cat `~/.python-version`).

So, now:
```sh
python --version
pyenv version
```
should output `3.9.12`

In the same way:
```sh
pyenv versions
```
should output something similar to:
```
  system
  3.6.0
* 3.9.12 (set by /Users/fperez/dev/spanish-classifier-tfg/.python-version)
  anaconda3-4.3.0
```

Now the problem with Python versions should be solved in your Mac! If you start a new terminal,
the `python --version` command should show `3.9.12`. Check it out by opening a new terminal!

Find more info about pyenv [here](https://realpython.com/intro-to-pyenv/)


### **Install [Poetry](https://python-poetry.org/docs/)**

e.g.:

`curl -SSL https://install.python-poetry.org | python3 -`

Then, to make avalable the poetry command:

```sh
echo 'export PATH="${PATH}":"${HOME}/.local/bin"' >> ~/.zshrc
```

Restart the terminal!

When back, let's show the poetry version:
```sh
poetry --version
```
should show you something like `1.4.0`. Now poetry is working.

If problems installing Poetry, see [this](https://github.com/python-poetry/install.python-poetry.org/issues/52#issuecomment-1387062081).

Clone repo in your `$HOME/dev` directory:

```sh
cd $HOME/dev
git clone git@github.com:NFanlo/spanish-classifier-tfg.git
```

Move to the repo:

`cd spanish-classifier-tfg`

Activate the shell, which in turn will create a Python virtual environment for the project.
```
poetry shell
```

Check the current python used is the one in the `.venv` create by poetry:
```sh
which python
```
should show something like: `/Users/fperez/dev/spanish-classifier-tfg/.venv/bin/python`

Install the project dependencies:
```sh
poetry install --with dev
```

All the dependencies for the project will be included in the `.venv` created by the command above.

```sh
poetry env list
```
should show something like:
```
.venv (Activated)
```

### VSCode with poetry

Once the project has been configured with the virtual environment, you can use it (and you should) also in VSCode.

Steps:

1. Close VSCode (just in case you had the project window already open)
2. In the project directory (e.g. `~/dev/spanish-classifier-tfg`), execute the command:
```sh
code .
```

Then a new window for the project should appear in VSCode. When selecting a Python file it should show in the
lower right part of the IDE the Python version 3.9.12 and the active virtual environment ('.venv': poetry).

**NOTE:** If the `.venv` virtual enviroment is not activated by default, activate it manually in VSCode using `CMD+Shift+P` and selecting the Python interpreter from the `.venv` directory in the project.

Also if you open a terminal in VSCode, the virtual environment should be detected and something similar to the
following should appear in the terminal:
```
source /Users/fperez/dev/spanish-classifier-tfg/.venv/bin/activate
```

### **Linting**

Main considerations:

- 120 chars per line (setup your editor for that)
- Google style over pep8 (we can change it if we want to)

If you want to run linting before commit, stage files and then run:

`./project_tasks/lint.sh`

NOTE: If at some point some of the linting errors slows down our goal (e.g. typing with mypy or docstring) we can decide to relax the rules.

# Usage

## TL;DR

For the impatient, the following commands will 1) transform the 60-20-20 dataset (dirty), 2) launch a train experiment with the standard distilbert model for 4 epoch and 3) will do inference on the test set with the resulting model on the experiment of step 2). For varying the rest of the parameters in each script, see the remaining sections below.

```sh
# Create the dataset transformations for the dirty 60-20-20 split
./scripts/dataset.sh false
# Execute an experiment training and evaluating the model on the train/dev set transformations created in the previous step
./scripts/train.sh distil
# Infer using the resulting finetuned model on step 2)
./scripts/infer.sh ep_4-lr_5e-5-msl_72-bs_8-ds_config_60-20-20-nl_6-do_0.1 distil
```

## Transformed Dataset Creation

Allows to move from the raw datasets in the `/dataset` directory (e.g. `60-20-20/` split) to a properly transformed dataset ready to be used in the `${TEMPDIR}/<split>` dir.
This transformed dataset will be used by the `train.sh` and `infer.sh` scripts.

The shell script to transform datasets is `./scripts/dataset.sh` which requires a single argument to control if using a cached dataset with the values `true` or `false`. The rest of the parameters of the script are controlled with environment variables passed before the command. Check the script `./scripts/dataset.sh` to see which options you have.


For example the command:
```sh
CLEAN_DS=true DS_CONFIG=60-20-20 ./scripts/dataset.sh false
```
as the output shows, this commnad will create the directory `$TMPDIR/60-20-20-cleaned` which contains the split data for the dataset applying the cleaning functions. Check it's contents with:
```sh
ls $TMPDIR/60-20-20-cleaned
```

And the command:
```sh
CLEAN_DS=false DS_CONFIG=80-10-10 ./scripts/dataset.sh false
```
as the output shows, this commnad will create the directory `$TMPDIR/80-10-10` which contains the split data for the dirty dataset. Check it's contents with:
```sh
ls $TMPDIR/80-10-10
```

## Train

The shell script to train models is `./scripts/train.sh` which requires a single argument to control which model to use. The values of the model are:
1. `distil` the regular distilbert model
2. `distilmulti` the distilbert multilingual
3. `distilbeto` the distilbeto model
4. `distilbetomldoc` the distilbeto finetuned with mldoc dataset

If something different of those names are used, it will train the regular distilbert model.

The rest of the parameters of the script are controlled with environment variables passed before the command. Check the script `./scripts/train.sh` to see which options you have.

For example the command:
```sh
CLEAN_DS=true DS_CONFIG=60-20-20 EPOCHS=2 ./scripts/train.sh distil
```

Will train the regular distilbert model during 2 epochs using the 60-20-20 split of the clean dataset (which is expected in the directory output of the previous command `./scripts/dataset.sh` which if hasn't been modified will be `$TMPDIR/60-20-20-cleaned`)

For the previous command, the model checkpoints will be stored in (check the output of the training process): `$HOME/dev/data/spanishclassfier_exp/distilbert-base-uncased-finetuned-with-spanish-tweets-clf-cleaned-ds/ep_2-lr_5e-5-msl_72-bs_8-ds_config_60-20-20-nl_6-do_0.1/`

The best checkpoint will be stored in the subdirectory `best_model`.

And the command:
```sh
CLEAN_DS=false DS_CONFIG=80-10-10 EPOCHS=2 DROPOUT=0.2 DISTIL_LAYERS=5 ./scripts/train.sh distilbeto
```
Will train the distilbeto model during 2 epochs with only 5 layers (instead of the 6 by default) and a dropout of 0.2 using the 80-10-10 split of the dirty dataset (which is expected in the directory output of the previous command `./scripts/dataset.sh` which should be `$TMPDIR/80-10-10`)

For the previous command, the model checkpoints will be stored in (check the output of the training process): `$HOME/dev/data/spanishclassfier_exp/dccuchile-distilbert-base-spanish-uncased-finetuned-with-spanish-tweets-clf/ep_2-lr_5e-5-msl_72-bs_8-ds_config_60-20-20-nl_5-do_0.2/`

Again, the best checkpoint will be stored in the subdirectory `best_model`.

See the `train.sh` script to see what are the options available! There are plenty!!!

**NOTE:** For early stopping, the default criteria is evaluate the model every 100 steps (`--evaluation_strategy` and `--eval_steps`) on the f1 metric (`--metric_for_best_model`), saving it also every 100 steps (`--save_strategy` and `--save_steps`) and with a patience of 10 (`--early_stopping_patience`). Only the best two checkpoints are saved for space reasons (`--save_total_limit`). Modify these paraaeters in the script at your will by consulting [the huggingface documentation on the trainer and training arguments in particular](https://huggingface.co/docs/transformers/v4.27.1/en/main_classes/trainer#transformers.TrainingArguments)

## Infer

The shell script to train models is `./scripts/infer.sh` which requires 2 arguments:
1. Experiment id to use (relative to the `$HOME/dev/data/spanishclassfier_exp/` directory)
2. Model id to use (distil, distilmulti, distilbeto or distilbetomldoc)

For example:

```sh
CLEAN_DS=true DS_CONFIG=60-20-20 ./scripts/infer.sh ep_2-lr_5e-5-msl_72-bs_8-ds_config_60-20-20-nl_6-do_0.1 distil
```

Will do inference with the test set (cleaned) with the model trained and stored in the experiment `ep_2-lr_5e-5-msl_72-bs_8-ds_config_60-20-20-nl_6-do_0.1` in the directory `$HOME/dev/data/spanishclassfier_exp/distilbert-base-uncased-finetuned-with-spanish-tweets-clf-cleaned-ds`.


And...

```sh
CLEAN_DS=false DS_CONFIG=80-10-10 ./scripts/infer.sh ep_2-lr_5e-5-msl_72-bs_8-ds_config_80-10-10-nl_5-do_0.2 distil
```

Will do inference with the test set (dirty) with the model trained and stored in the experiment `ep_2-lr_5e-5-msl_72-bs_8-ds_config_80-10-10-nl_5-do_0.2` in the directory `$HOME/dev/data/spanishclassfier_exp/dccuchile-distilbert-base-spanish-uncased-finetuned-with-spanish-tweets-clf`.
