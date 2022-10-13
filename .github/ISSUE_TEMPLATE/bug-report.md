---
name: Bug Report
about: The following are just suggestions for you to fill-in and make our work easier.
  The more info you provide, the higher the chances that we can solve the issue.
title: ''
labels: ''
assignees: ''

---

**mdciao version**
What version of mdciao are you using? 
A pip release or a github clone? In case of the latter, you can use 
```git log -n 1 --oneline```
to let us know what's the status of your local clone of mdciao.

**Describe the bug or unexpected behavior**
What happened?

**To Reproduce**
Please paste the code, either directly or as an attached python script or Jupyter notebook.

**Expected Behavior**
What did you expect to happen?

**Unexpected Behavior** 
Please paste the outputs needed to understand the suspected bug.

Tracebacks, text outputs or screenshots/plots if needed.

**Needed Data** 
If possible, upload the files needed run the code you just pasted. In case of molecular dynamics trajectories, paste a strided down version, just a couple of frames.

**Additional Context**
Add any other context about the problem here.

**OS**
What MacOS or GNU/Linux are you running?

**Python Specific Information**
What version of python are you running? 

Could you please attach your environment file:

For [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-the-environment-yml-file):
```conda env export > environment.yml```
For [pip](https://pip.pypa.io/en/stable/cli/pip_freeze/#examples):
```python -m pip freeze > requirements.txt```
