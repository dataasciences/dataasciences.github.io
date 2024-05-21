---
title: "Lesser-Known Jupyter Magic Commands"
excerpt: "Handy Jupyter Magic Commands"
categories:
  - Jupyter Notebook
  - Python 
tags:
  - Data Science
  - Python
  - Jupyter Notebook
sidebar:
  - nav: docs
  - title: "BUY ME COFFEE!"
    image: "/assets/images/buy_me.jpg"
    image_alt: "image"
---

The Space of Data Science has several tools that can be used for Data Extraction, Model building, Deployments & Reporting. But one tool or an IDE that is commonly used across is Jupyter Notebooks. Knowing handy commands and tricks in Jupyter Notebook can be helpful when you develop your projects, debug them, or when there is a need to export your project as a module, and many more.

In this blog, we will see 10 lesser-known Jupyter commands with examples.

### Magic Commands

```python
%who
```

This magic command displays all interactive variables in your namespace. It's helpful for checking which variables are currently defined.

Check the image below that I executed in my local Jupyter instance.  

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_who.PNG?raw=true)

The namespace is empty as I have no variables. Here is the output after I declare some variables.

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_who1.PNG?raw=true)


```python
%reset
```

reset will clear your namespace without restarting the kernel. Adding the -f flag will force the reset without asking for confirmation.

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_reset.PNG?raw=true)


```python
%%timeit
```

This command times the execution of a Python statement or expression. It's useful for quick performance comparisons.

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_timeit.PNG?raw=true)

> <span style="font-size:1em;"> **Purpose:** %timeit is primarily used for timing small code snippets or expressions repeatedly to get a more accurate measure of their execution time.<br>
                  **How it works:** It runs the specified code multiple times (by default, it runs it 100,000 times) and calculates the average execution time, providing a more reliable estimate.<br>
                  **Output:** %timeit provides a more detailed output, including the average time taken per loop, the number of loops executed, and the best time taken per loop.
                  </span>
{: .notice--info}

%timeit is useful for benchmarking small code snippets and obtaining reliable average execution times.

```python
%history
```

Want to see the history of your commands? %history displays a list of previous inputs along with their line numbers. Here is an example for the instance i ran on,

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_history.PNG?raw=true)


```python
%pdb
```

This command activates the Python debugger (pdb) automatically whenever an exception is raised. It allows for interactive debugging right in the notebook.

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_pdb.PNG?raw=true)

Once the pdb debugger is activated, it will be of assistance whenever an exception arises and debugging becomes easier, as shown below;

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_pdb1.PNG?raw=true)


```python
%env
```

With **%env**, you can view and modify environment variables within the notebook. It's particularly useful for managing configurations and dependencies.

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_env.PNG?raw=true)


```python
%notebook
```

**%notebook** command allows you to export the current notebook to a Python script file (.py). The .py file gets saved to the same path the jupyter notebooks get saved in. 

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_notebook_file.PNG?raw=true)

The above command saves the jupyter notebook content to the "jupyter.py" python module.


```python
%load_ext
```

If you have custom IPython extensions, you can load them into your notebook using %load_ext extension_name.

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_loadext.PNG?raw=true)

In the case above, I am using "autoreload" extension. It reloads modules automatically before entering the execution of code typed at the IPython prompt.
If you are interested to know more about this, refer to the [page here](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html).

```python
%time
```

Similar to %timeit, %time measures the execution time of a single Python statement or expression, but it only runs it once.

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_timeit.png?raw=true)

Based on the output above, here is what they mean; 

> <span style="font-size:1em;"> The **Wall Time** means that a clock hanging on a wall outside of the computer would measure 995 ms from the time the code was submitted to the CPU to the time when the process completed.<br>
             **User time** and **sys time** both refer to the time taken by the CPU to actually work on the code. The CPU time dedicated to our code is only a fraction of the wall time as the CPU swaps its attention from               our code to other processes that are running on the system.<br>
            **User time** is the amount of CPU time taken outside of the kernel. Sys time is the amount of time taken inside of the kernel. The total CPU time is user time + sys time.
  </span>
{: .notice--info}

```python
%xmode
```
**%xmode** changes the exception reporting mode in IPython. You can set it to "Plain", "Context", or "Verbose" depending on how much information you want when an exception occurs.

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/jupyter_xmode.PNG?raw=true)

I hope these Magic commands are useful to you as you continue to master Jupyter Notebooks.

**If you like the content, consider supporting me!**
{: .notice--info}

<!---
[![Support via PayPal](https://cdn.jsdelivr.net/gh/twolfson/paypal-github-button@1.0.0/dist/button.svg)](https://www.paypal.me/mmistakes)
{: style="margin-top: 0.5em;"}
-->

**You could get me a coffee!** 

[!["Buy Me A Coffee"](https://user-images.githubusercontent.com/1376749/120938564-50c59780-c6e1-11eb-814f-22a0399623c5.png)](https://buymeacoffee.com/softwaremusings)

