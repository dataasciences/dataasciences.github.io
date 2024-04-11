---
title: "Ohh, Big-O!"
excerpt: "Basics of Big-O"
categories:
  - Algorithms
  - Data Structures
  - Programming
tags:
  - Programming
---

If you attend FAANG/MAANG interviews, chances are you would be asked to calculate the O-Notation for the code you write.

Any software or a program that is written would have to run efficiently, otherwise chances are no one is going to use what you build. For this, they must go through 
algorithm analysis that analyzes the complexity of different algorithms and finds the most efficient algorithm to solve the problem at hand. Here is where Big-O Notation comes to the rescue. One could argue that time taken is an easier approach to analyse an algorithms' efficiency, but relatively speaking it can't be used as a metric to decide as it entirely depends on the hardware the program is being executed on.

We will look at what Big-O Notation is and some sample code in Python along with its corresponding Big-O. 

**Note:** Big-O notation is one of the measures used for algorithmic complexity. Some others include Big-Theta and Big-Omega. Big-Omega, Big-Theta, and Big-O are intuitively equal to the best, average, and worst time complexity an algorithm can achieve. We typically use Big-O as a measure, instead of the other two, because it can guarantee that an algorithm runs in an acceptable complexity in its worst case, it'll work in the average and best case as well, but not vice versa.
{: .notice--warning}

## What is Big-O Notation?

Big-O notation is a statistical measure used to describe the complexity of the algorithm. Big-O describes an algorithm's runtime or memory consumption (space) without the interference of contextual variables such as RAM and CPU. It gives programmers a way to compare algorithms and identify the most efficient solution. It signifies the relationship between the input to the algorithm and the steps required to execute the algorithm.

**Note:**
Time-Complexity: The Number of steps required to complete the execution of an algorithm
Space-Complexity: Amount of space you need to allocate in memory during the execution of a program.
{: .notice--warning}

**𝗕𝗶𝗴 𝗢 𝗮𝗻𝘀𝘄𝗲𝗿𝘀 𝗼𝗻𝗲 𝘀𝘁𝗿𝗮𝗶𝗴𝗵𝘁𝗳𝗼𝗿𝘄𝗮𝗿𝗱 𝗾𝘂𝗲𝘀𝘁𝗶𝗼𝗻:**

    🔸 How much does runtime or memory consumption grow as the size of the input increases, in the worst-case scenario?

To compute Big-O, you will first need to know the possible values it can consist of. It is common for both Time & Space Complexity, however, time complexity is what is measured more commonly between the two.

| Notation        | Name           | Description  |
| ------------- |:-------------:| -----:|
| O(1)        | Constant Complexity     | The time & space consumed is consistent regardless of input size |
| O(log n)    | Logarithmic Complexity  | Time & Space consumed grows logarithmically with Input Size |
| O(n)        | Linear Complexity       | Time & Space consumed grows in direct proportion to the Input Size |
| O(n log n)  | Linearithmic Complexity | Time & Space consumed grows proportionally to n log n (where n is the size of the input) |
| O(n^2)      | Quadratic Complexity    | Time & Space consumed grows with the Square of the Input Size |
| O(n^3)      | Cube Complexity         | Time & Space consumed grows with the Cube of the Input Size |
| O(2^n)      | Exponential Complexity  | Time & Space consumed doubles with each increment to the Input Size |
| O(n!)       | Factorial Complexity    | Time & Space consumed grows factorially to the Input Size |

**Note:** Anything worse than linear is considered a bad complexity (i.e. inefficient) and should be avoided if possible. Linear complexity is okay and usually a necessary evil. Logarithmic is good. Constant is amazing!
{: .notice--warning}

## Some Examples:

Let's say we want to calculate the Mean of a sequence, a naive implementation would be as follows;

```python
vals = [1, 2, 3, 4]

def mean(lst):
  total = 0
  size = 0
  for val in lst:
    total += val
    size += 1
  return total / size
```

Can you guess the Big-O value for the above function?

We have to loop over every item in the sequence. As there are N items, the runtime complexity of this function is O(N).
Let's look at another example and compare it with an efficient approach. 

```python
names = ['Andy', 'Sarah', 'Bob', 'David', 'Chris']

def find_values(lst, val):
  for item in lst:
    if item == val:
      rt = 'Found it!!'
    else:
      rt = 'NO!'
  return rt

find_values(names, 'Sarah')
```
The above function is considered O(N) because it is looping over the list fully although the value is present in index 1 of the input list.
Now, we can take advantage of the order and find items quicker. A binary search algorithm does not look at each item but finds a central point, and if the item is less than the centre, it only considers half of the items. It repeats this process. Here is the python function for that.

```python
names = ['Andy', 'Sarah', 'Bob', 'David', 'Chris']

def sorted_search(lst, val):
  lst = sorted(lst)
  id_value = int(len(lst) // 2)
  if lst[id_value] == val:
    return 'Found it!!'
  elif lst[id_value] > val and len(lst) > 1:
    return sorted_search(lst[:id_value], val)
  elif lst[id_value] < val and len(lst) > 1:
    return sorted_search(lst[id_value:], val)
  else:
    return 'Not Found'
```

On average, Binary Sorted Search only looks at 19 items for lists of size around 1,000,000. The above function has a Big-O of O(logN). It only looks at the log of the number of items. Logarithmic complexity is desirable, as it achieves good performance even with highly scaled input.

Here is a plot of common Big-Os. Once your data gets big, the runtime complexity of your algorithm can have a large effect on how long your computations take.

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/GIuFHDtXcAEoqcQ.jpeg?raw=true)

## Worst vs Best Case Complexity

Usually, when someone asks you about the complexity of an algorithm - they're interested in the worst-case complexity (Big-O). Sometimes, they might be interested in the best-case complexity as well (Big-Omega). To understand the relationship between these two, let's take a look at another piece of code; 

```python
def search_algo(num, items):
    for item in items:
        if item == num:
            return True
        else:
            pass
nums = [2, 4, 6, 8, 10]

print(search_algo(2, nums))
```

In the code above, we have a function that takes a number and a list of numbers as input. It returns true if the passed number is found in the list of numbers, otherwise, it returns None. If you search for 2 in the list, it will be found in the first comparison. This is the best-case complexity of the algorithm in that the searched item is found in the first searched index. The best case complexity, in this case, is O(1). 

On the other hand, if you search 10, it will be found in the last searched index. The algorithm will have to search through all the items in the list, hence the worst-case complexity becomes O(n).

Now that you know how to compute Big-O, here are a few scenarios where Big-O can be used:
    
    🔸 Live coding challenges
    🔸 Code walk-throughs
    🔸 Discussions about projects/solutions you've built

Finally, you ought to compute it when you come up with a solution or compare an algorithm with another. The ability to consider a solution's viability beyond whether it works or not shows maturity in your decision-making and approach to programming.
