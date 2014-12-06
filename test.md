# Visualizing Data

Hi kyle!

This chapter outlines techniques for data visualization in Python.
For many years, the leading package for scientific visualization has been Matplotlib.
Though recent years have seen the rise of other tools -- most notably Bokeh http://bokeh.pydata.org/ for browser-based interactive visualization, and VisPy http://vispy.org/ for high-performance visualization based on OpenGL -- matplotlib remains the leading package for the production of publication-quality scientific visualizations.
Documentation for matplotlib can be found at http://matplotlib.org/.


##### Plotting from a script


If you are using matplotlib from within a script, the function ``plt.show()`` is your friend.
``plt.show()`` starts an event loop, looks for all currently active figure objects, and opens interactive windows which display your figure or figures.

So, for example, you may have a file called ``myplot.py`` which contains the following:

<pre data-executable="ipython" data-code-language="python">
%pylab inline
# ------- file: myplot.py ------
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x*5) + np.cos(x))

plt.show()
</pre>

You can then run this script from the command-line prompt:

```
$ python myplot.py
```

and this will result in a window opening with your figure.
The ``plt.show()`` command does a lot under the hood, as it must interact with your system's interactive graphical backend. The details of this operation can vary greatly from system to system and even installation to installation, but matplotlib does its best to hide all these details from you.

One thing to be aware of: the ``plt.show()`` command should be used *only once*, most often at the end of your script. Multiple ``show()`` commands can lead to unpredictable backend-dependent behavior, and should mostly be avoided.

##### Plotting from an IPython shell


It can be very convenient to use IPython interactively from within the terminal.
If your system is correctly set up, you can start IPython by running the ``ipython`` command from the prompt:

```
$ ipython
```

The result is an interactive Python prompt where you can execute statements line-by-line, and view the results interactively.
The prompt line numbers will be labeled, e.g. ``In [1]:``

The best way to use matplotlib from within IPython is to use the ``matplotlib`` mode, which can be activated using the ``%matplotlib`` magic command after starting ``ipython``:

``` ipython
In [1]: %matplotlib
Using matplotlib backend: TkAgg
```

At this point, any ``pyplot`` plot command will cause a figure window to open, and further commands can be run to update the plot. Some changes (such as modifying properties of lines that are already drawn) will not draw automatically: to force an update, use ``plt.draw()``.

##### Plotting from an IPython Notebook


The IPython notebook is a browser-based interactive data analysis tool (see chaper X.X) which can combine narrative, code, graphics, HTML elements, and much more into a single executable document.
It can be started from the command prompt,

```
$ ipython notebook
```

at which point a browser window will open to a local server and allow viewing and creation of notebooks.

Plotting interactively within an IPython notebook can be done with the ``%matplotlib`` command, and works similarly to the IPython shell, above. In the IPython notebook, you also have the option of embedding graphics directly in the notebook. This can be done by adding ``inline`` to the magic command:

``` ipython
In [1]: %matplotlib inline
```

After running this command (it need only be done once per session), any cell within the notebook which creates a plot will embed a png image of the resulting graphic:


``` python
%matplotlib inline
```



<pre data-executable="ipython" data-code-language="python">
%pylab inline
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x));
</pre>


#### Saving Figures to File


One nice feature of matplotlib is the ability to save figures in a wide variety of formats.
Saving a figure can be done using the ``plt.savefig`` command.
For example, to save the previous figure as a pdf file, you can run


``` python
plt.savefig('my_figure.pdf')
```


The file format is inferred from the extension of the given filename.
Depending on what backends you have installed, many different file formats are available.
These can be found for your system by using the following method of the ``canvas`` object:


``` python
plt.figure().canvas.get_supported_filetypes()
```


Note that when saving your figure, you need not use ``plt.show()`` or related commands discussed above.

#### Learning More about ``matplotlib``


A single chapter in a book can never hope to cover all the available features and plot types available in matplotlib.
In addition to the help routines described in section X.X, an extremely helpful reference is matplotlib's online documentation at http://matplotlib.org/.
In particular see the Gallery linked on that page: this shows thumbnails of hundreds of different plot types, each one linked to a page with the Python code snippet used to generate it.
In this way, you can visually inspect and learn about a wide range of different visualization techniques.

### Sidebar: Two Interfaces for the Price of One


A potentially confusing feature of matplotlib is its dual interfaces: a MatLab-style state-based interface, and a more powerful object-oriented interface. We'll quickly highlight the differences between the two here.

##### MatLab-style Interface


Matplotlib was originally written as a Python alternative for MatLab users, and much of its syntax is reflected by that fact.
The MatLab-style tools are contained in the ``pyplot`` interface, which we imported above under the standard appreviation ``plt``.  The pyplot interface is a state-based interface: that is, matplotlib keeps track of a current axes object and runs commands on that. When a new axes object is created, that new object becomes current:


``` python
plt.figure()

# create the first of 2 panels & select it
# (2, 1 gives the shape of the grid,
#  and 1 indicates the first panel)
plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x))

# create the second panel & select it
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x));
```


This interface is *stateful*: it keeps track of the "current" axes and figure, which is where all ``plt`` commands are applied.
You can get a reference to these using the ``plt.gca()`` (get current axes) and ``plt.gcf()`` (get current figure) routines.

While this stateful interface is fast and convenient for simple plots, it is easy to run into problems.
For example, once the second panel is created, how can we go back and add something to the first?

##### Object-Oriented Interface


The Object-oriented interface is available for these more complicated situations.
Rather than depending on some notion of an "active" figure or axes, in the object-oriented interface the plotting functions are *methods* of ``Figure`` and ``Axes`` objects.
To recreate the above plot using this style of plotting, you might do the following:


``` python
# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)

# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x));
```


Which style to use is largely a matter of taste for more simple plots, but the object-oriented approach can become a necessity as plots become more complicated.
Throughout this chapter, we will switch between the matlab-style and object-oriented interfaces, depending on what is most convenient.
In most cases, the difference is as small as switching ``plt.plot`` to ``ax.plot``, but there are a few gotchas that we will highlight in the following pages.



``` python
# HIDDEN
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
```


