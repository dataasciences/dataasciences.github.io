---
title: "What changed in Python 3.8 & above?"
excerpt: "Differences and upgrades in Python 3.8 & above"
categories:
  - Python 
tags:
  - Python
  - Software Development
sidebar:
  - nav: docs
  - text: >
      [![](/assets/images/buy_me_coffee.jpg)](https://buymeacoffee.com/softwaremusings){:target="_blank"}
classes: wide
---

Evolution is not just for us, Homo Sapiens, but also applies to Software (No, AGI is not here yet!) and Python is no exception.
Python as a programming language has evolved into various versions and updates over the years, right from 2.x to now 3.12.

But among all these upgrades, python improved marginally in 3.8 over 3.7 in some areas and added new syntax, and speed improvements and still maintained the tradition from the earlier versions. 

If you are using Python 3.8 and are looking to improve on version < 3.7, here are the key changes made in this version;

### 1. Assignment Expressions (The Walrus Operator)

One of the most talked-about features is the introduction of the assignment expression operator (:=), often referred to as the "walrus operator". It allows assignments within expressions, enabling assignments to be made as part of larger expressions.

```python
# Before Python 3.8
import re

pattern = re.compile(r'\d+')
text = "123 abc 456 def"

match = pattern.search(text)
if match:
    print(f"Found number: {match.group(0)}")

# Python 3.8 and later
import re

pattern = re.compile(r'\d+')
text = "123 abc 456 def"

if match := pattern.search(text):
    print(f"Found number: {match.group(0)}")    
```
### 2. Positional-only Parameters

Python 3.8 introduced support for defining positional-only parameters in function definitions using the / separator. This allows developers to enforce the positional-only behavior for certain parameters, providing more control over function signatures.

A special marker, /, can now be used when defining a method's arguments to specify that the function only accepts positional arguments on the left of the marker.

```python
def func(a, b, /, c, d):
    # a and b are positional-only parameters
    # c and d can be positional or keyword arguments
    pass
```

Example for positional-only parameter;

```python
def greet(name, /, greeting="Hello"):
    return f"{greeting}, {name}!"

# Usage examples:
print(greet("Alice"))          # Correct usage: name as positional argument
print(greet("Bob", "Hi"))      # Correct usage: name and greeting as positional arguments

# Incorrect usage:
# print(greet(name="Alice"))   # Raises TypeError: name must be a positional argument

# correct usage
print(greet("Charlie", greeting="Hey"))  # Correct usage: name as positional, greeting as keyword
```

* 'name' is a positional-only parameter. It must be provided as a positional argument and cannot be given as a keyword argument.
* 'greeting' is a regular parameter that can be provided either positionally or as a keyword argument.

**Correct Usage:**

* greet("Alice"): name is provided positionally.
* greet("Bob", "Hi"): Both name and greeting are provided positionally.
* greet("Charlie", greeting="Hey"): name is positional, greeting is provided as a keyword argument.

**Incorrect Usage:**

* greet(name="Alice"): Raises a TypeError because 'name' must be positional.

This feature provides more control over the function's interface, ensuring certain parameters are always passed positionally, which can help avoid common mistakes and improve code readability.

### 3. f-strings Support = for Self-documenting Expressions

Python 3.8 extended f-strings to support the = specifier, which formats the expression and its value in a way that makes it easier to debug and understand.
This feature is particularly useful for debugging and logging, as it allows you to easily see the value of an expression along with its name.

Here's an example demonstrating the use of the = specifier with f-strings:

```python
# Python 3.8 and later
name = "Alice"
age = 30
height = 165.5

# Using f-strings with the = specifier
print(f'{name=}')
print(f'{age=}')
print(f'{height=:.2f}')  # You can still use format specifiers

# Multiple expressions
print(f'{name=}, {age=}, {height=:.2f}')

## OUTPUT

#name='Alice'
#age=30
#height=165.50
#name='Alice', age=30, height=165.50
```
**Single Expression:**

* f'{name=}' outputs name='Alice', showing both the expression name and its value 'Alice'.
* f'{age=}' outputs age=30, showing both the expression age and its value 30.
* f'{height=:.2f}' outputs height=165.50, showing both the expression height and its value 165.50 formatted to two decimal places.

**Multiple Expressions:**

* f'{name=}, {age=}, {height=:.2f}' combines multiple expressions, outputting name='Alice', age=30, height=165.50, making it easy to see the values of multiple variables in a single print statement.  

Using the = specifier in f-strings simplifies debugging and logging by automatically including both the variable name and its value, reducing the need for repetitive and verbose code.

### 4. TypedDict

In Python 3.8, TypedDict was introduced as part of the typing module, providing a way to define dictionary types with a fixed set of keys and associated value types. This enhancement allows for more precise type checking, making it easier to define and enforce the structure of dictionaries in your code.

```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email: str

# Example usage
person: Person = {
    'name': 'Alice',
    'age': 30,
    'email': 'alice@example.com'
}

# Type checker will raise an error if a required key is missing or if a value is of incorrect type
```

**Defining the TypedDict:**

* class Person(TypedDict): defines a dictionary type Person with specific keys and their associated types.
* name, age, and email are defined as keys with types str, int, and str respectively.

**Using the TypedDict:**

* person: Person declares that person is of type Person.
* The dictionary person must have the keys name, age, and email with the corresponding types specified.

**Key Features of TypedDict in Python 3.8**

*Fixed Set of Keys:*
- 'TypedDict' allows you to define a dictionary with a fixed set of keys, improving type safety by ensuring only the specified keys are present.

*Optional Keys:*
- TypedDict supports optional keys, allowing you to specify that certain keys are not required.

```python
from typing import TypedDict, Optional

class Person(TypedDict):
    name: str
    age: int
    email: Optional[str]  # email is optional

# Example usage with an optional key
person: Person = {
    'name': 'Alice',
    'age': 30,
    # 'email' key can be omitted
}
```  

*Inheriting TypedDict:*
- You can create more specific TypedDict types by inheriting from existing TypedDict types, enabling reusable and extendable dictionary structures.

```python
class Employee(Person):
    employee_id: int

# Example usage of an inherited TypedDict

employee: Employee = {
    'name': 'Bob',
    'age': 25,
    'email': 'bob@example.com',
    'employee_id': 1234
}
```

TypedDict in Python 3.8 provides a powerful way to define the structure and types of dictionaries, enabling better type checking and reducing runtime errors. It enhances code readability and maintainability by explicitly specifying the expected dictionary format.

### 5. New Syntax Warnings and Error Handling Improvements:

Python 3.8 introduced several new syntax warnings and improvements to error handling that help developers catch potential issues earlier and understand error messages more clearly. Here are some of the key enhancements.

**Syntax Warnings:**

*1. Deprecation Warnings for Outdated Syntax*
    
- Python 3.8 issues warnings for deprecated or outdated syntax that may be removed in future versions. This helps developers to update their code proactively.

*2. Future Warnings for Incompatible Changes*

- Python 3.8 introduces FutureWarning for features that are planned to change in future releases. This allows developers to prepare their codebases for upcoming changes.

**Error Handling Improvements:**

*1. Improved SyntaxError Messages*

- Python 3.8 provides more informative SyntaxError messages, making it easier to understand what caused the error and how to fix it. This includes better highlighting of the error location within the code.

*2. More Detailed Exception Tracebacks*

- Tracebacks in Python 3.8 include additional context to help diagnose errors. For example, KeyError now displays the missing key, and IndexError shows the out-of-range index.  

**Detailed KeyError and IndexError Messages:**

```python
# Example of KeyError with improved message
my_dict = {"name": "Alice"}
print(my_dict["age"])
# Output: KeyError: 'age'
# The KeyError now explicitly shows the missing key.

# Example of IndexError with improved message
my_list = [1, 2, 3]
print(my_list[5])
# Output: IndexError: list index out of range
# The IndexError message now makes it clear that the index is out of the valid range.
```
**New Syntax Warnings Example:**

Python 3.8 introduces new warnings to alert developers to potential issues in their code. One such warning is the **SyntaxWarning** for using the **is operator** with literals, which is often a mistake.

```python
# Example of SyntaxWarning
x = 10
if x is 10:
    print("x is 10")
# Output: SyntaxWarning: "is" with a literal. Did you mean "=="?
# The warning suggests that `==` should be used instead of `is` with literals.
```
**Deprecation Warnings**

Deprecation warnings help developers identify and update code that uses features slated for removal in future Python versions.

```python
import warnings

warnings.warn("This is a deprecated feature", DeprecationWarning)
# Output: DeprecationWarning: This is a deprecated feature
```

**Future Warnings**

Future warnings are used to signal upcoming changes that will affect compatibility in future versions.

```python
import warnings

warnings.warn("This will change in the future", FutureWarning)
# Output: FutureWarning: This will change in the future
```

Python 3.8's new syntax warnings and error handling improvements enhance code quality and developer productivity by providing clearer and more actionable feedback. These changes help developers catch potential issues early, understand errors more easily, and prepare for future updates to the Python language.

And finally, 
### 6. Performance Improvements and Optimizations:

Python 3.8 introduced several performance improvements and optimizations that enhance the overall efficiency of the language. These enhancements cover various aspects of the language, including function calls, memory usage, and standard library optimizations. Here are some key performance improvements and optimizations in Python 3.8:

*1. Function Call Optimizations*

- Positional-Only Parameters: The introduction of positional-only parameters allows for faster function calls because the interpreter can optimize how arguments are passed and validated.

- Vectorcall Protocol: Python 3.8 introduces a new calling convention for CPython, known as vectorcall, which reduces the overhead of function calls by using arrays of arguments. This results in faster calls to certain     built-in functions and methods.

*2. Memory Usage Improvements*
  
- Pycache Files: Python 3.8 reduces the size of .pyc files by avoiding the storage of unused variables and constants, which leads to better memory efficiency and faster loading times.
  
- Shared Keys Dictionaries: Dictionaries with identical keys share a common key object, reducing memory usage and improving performance when creating many similar dictionaries.

*3. Optimized Standard Library*
  
- Improved Modules: Various standard library modules have been optimized for better performance. For instance, the math module has received several optimizations for mathematical operations.
  
- f-string Performance: Formatting strings using f-strings is faster in Python 3.8 compared to previous versions, making it more efficient to use for string formatting tasks.  

**Example: Memory Usage Optimization with Shared Keys**

```python
# Shared keys dictionaries
dict1 = {'name': 'Alice', 'age': 30}
dict2 = {'name': 'Bob', 'age': 25}

# Both dict1 and dict2 share the same key object internally, saving memory
```

Python 3.8 brings a variety of performance improvements and optimizations that make the language faster and more efficient. These enhancements benefit a wide range of applications, from those requiring quick startup times to those dealing with heavy computational tasks. The introduction of positional-only parameters, the vectorcall protocol, and various memory and standard library optimizations collectively contribute to a more performant Python.

These are some of the improvements over version 3.7 and earlier versions. It's beneficial to consider these updates when you develop your applications to see an overall performance improvement of your application.

**If you like the content, consider supporting me!**
{: .notice--info}

**You could get me a coffee!** 

[!["Buy Me A Coffee"](https://user-images.githubusercontent.com/1376749/120938564-50c59780-c6e1-11eb-814f-22a0399623c5.png)](https://buymeacoffee.com/softwaremusings)

