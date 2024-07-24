---
title: "Python - Tips & Hacks"
permalink: /pythontips/
sidebar:
  - nav: docs
classes: wide
---

## Listing Outdate Packages in Python 

Python3 comes with a list of pre-installed packages which are updated from time to time. It is really easy to list any *outdated* package with the following command:

```python
pip list --outdated

# Output
Package    Version   Latest     Type
---------- --------- ---------- -----
astroid    2.0.4     2.1.0      wheel
certifi    2018.4.16 2018.11.29 wheel
pylint     2.1.1     2.2.2      wheel
setuptools 40.0.0    40.6.3     wheel
six        1.11.0    1.12.0     wheel
```

As you can see 5 packages on my machine are outdated.

To update a specific package to the latest version you can use

```python
pip install --upgrade  <PackageName>

# Alternatively you can use the short form
pip install -U <PackageName>
```

In you don't want to install each update individually you can use *pipdate*

```python
# Install pipdate
sudo pip3 install pipdate

# Update all packages
sudo pipdate3
```

**Note for Windows users:** *Pipdate* is already installed by default so you don't need to install it

**Note for Mac users:** If you installed *Python* via **homebrew** there is no need to use *sudo* to install *pipdate* or upgrade packages
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## File Navigations

This is something I used to stumble on a lot in the early days of my Python coding adventure so I thought to write a quick article about it.

#### Python Open file

Opening a file for read or write in Python can be achieved with the following code:

```python
# Open file
file = open("/Python/Files/MyFile.txt")
```

#### Python Read file

To read content of the file we simply use the **read()** method

```python
# Read file to the end of file
file.read()

'This is the first line\nAnd a second\nAnd even a third\nShall we put a fourth?\nWhy not a fifth\nOr a sixt\n'
```

Note that Python will print the *newline* character and will not split lines for you (but that's material for another day/post)

#### Where is my text?

Say you want to print again file content issue the same command again will leave you with something like this

```python
# Read file again
file.read()
''
```

Not a lot to see on screen, so where did our file's content go?

#### Python seek()

When Python reads a file's content it will move *file current position* to the end of the file so trying to re-read it again will yield the above result. Nothing as there is nothing else to read.

You can easily go back to the beginning of the file with the *seek()* method which is used like this:

```python
# Go back to position 0
# Or beginning of file
file.seek(0, 0)
```

* **Syntax of seek() method** - fileObject.seek(offset[, whence])

* **offset** is the position of the read/write pointer within the file.

* **whence** is optional and defaults to 0 which means absolute file positioning, other possible values are 1 which means seek relative to the current position and 2 which means seek relative to the file's end
{: .notice--info}

Now if you try to read the file again all content will be correctly displayed

```python
file.read()

'This is the first line\nAnd a second\nAnd even a third\nShall we put a fourth?\nWhy not a fifth\nOr a sixt\n'
```
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🔍How to Create Infinite Iteration using "itertools.cycle"

Use-cases?

👉 Round-robin task scheduling.
👉Creating repeating patterns in data processing.
👉Continuously cycling through a list of configurations or parameters.

The itertools.cycle function can be used to iterate over a sequence infinitely.

👉 itertools.cycle: Takes an iterable and returns an iterator that produces the elements of the iterable in a cycle, repeating indefinitely.
👉 next: Used to get the next item from the iterator.

```python
import itertools

# Define a list of values
colours = ['red', 'green', 'blue']

# Create an infinite iterator
infinite_colors = itertools.cycle(colours)

# Use the iterator
for _ in range(10):
    print(next(infinite_colors))
```
