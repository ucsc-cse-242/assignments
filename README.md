# CSE 242 assignments

Use the supplied `Assignment1/Assignment1.py` starter. Fill the `# YOUR CODE HERE` function bodies and `# YOUR ANSWER HERE` comment blocks. Save the required plots from your Python program and follow the handout's submission instructions.

- Assignment 1: [handout](Assignment1/README.md), [Python interface](Assignment1/INTERFACE.md).

# Install the requirements
You will install the requirements listed in the repository's
[requirements.txt](requirements.txt) file.  To install these packages,
you will run the command:
```sh
pip3 install -r requirements.txt
```

# Installing the Autograder package
Install the autograder with `pip3 install autograder-py` on the command line.  

# Submitting and using the Autograder in CSE 240

Make sure that the autograder is installed on your local machine by
typing: `python3 -m autograder.cli`.  If you see the `--help` option:

```nil
python -m autograder.cli
The autograder CLI package contains several tools for interacting with the autograder.
The following is a non-exhaustive list of CLI tools.
Invoke each command with the `--help` option for more details.
```
## Using the autograder

The autograder command line interface (cli) is [documented](https://github.com/eriq-augustine/autograder-py).  As a
student in the class, the main commands you will use are:

-   `python3 -m autograder.run.submit`: this will submit an assignment
    for a particular class and assignment.
-   `python3 -m autograder.run.peek`: this will show you your last submission
-   `python3 -m autograder.run.history`: this will show a summary of all
    your past submission
