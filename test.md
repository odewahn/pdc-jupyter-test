# Visualizing Data



``` python
#HIDDEN
import numpy as np
```


## Visualizing Data


This chapter outlines techniques for data visualization in Python.
For many years, the leading package for scientific visualization has been Matplotlib.
Though recent years have seen the rise of other tools -- most notably Bokeh http://bokeh.pydata.org/ for browser-based interactive visualization, and VisPy http://vispy.org/ for high-performance visualization based on OpenGL -- matplotlib remains the leading package for the production of publication-quality scientific visualizations.
Documentation for matplotlib can be found at http://matplotlib.org/.

We will follow the standard convention and use the following abbreviations for matplotlib imports:


<pre data-executable="ipython" data-code-language="python">
%pylab inline
# Main Matplotlib package
import matplotlib as mpl

# Pyplot (i.e. pylab) interface
import matplotlib.pyplot as plt
</pre>


While the bulk of the recipes in this chapter will focus on matplotlib tools, in several recipes near the end of the chapter we will encounter some other visualization tools which you might find useful.

First, though, we'll look at some general information about how to use matplotlib effectively:

### A few ``matplotlib`` tips


Because matplotlib will be the general focus of this chapter, we'll start with some helpful general information about using the package.

#### ``show()`` or no ``show()``? How to Display your Plots


It goes without saying that a visualization must be seen to be useful.
Here we'll discuss quickly how to make sure you can see your visualization.
This often causes some confusion because there are two different contexts in which matplotlib is generally used either **in a script**, **in an IPython terminal**, or **in an IPython notebook**.

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

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

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


## Simple Line Plots


*This recipe covers the basics of setting up a matplotlib plot, and how to create simple line plots*

- ``plt.plot``
- ``plt.figure``
- ``plt.axes``
- ``plt.xlim``
- ``plt.ylim``
- ``plt.axis``
- ``plt.xlabel``
- ``plt.ylabel``
- ``plt.title``

Perhaps the simplest of all plots is the visualization of a single function $y = f(x)$.
Here we will take a first look at creating a simple plot of this type.
For all matplotlib plots, we start by creating a figure and an axes.
In their simplest form, a figure and axes can be created as follows:


``` python
fig = plt.figure()
ax = plt.axes()
```


In matplotlib, the *figure* (an instance of the class ``plt.Figure``) can be thought of as a single container which contains all the objects representing axes, graphics, text, labels, etc.
The *axes* (an instance of the class ``plt.Axes``) is what we see above: a bounding box with ticks and labels, which will eventually contain other plot elements.
Through this book, we'll commonly use the variable name ``fig`` to refer to a figure instance, and ``ax`` to refer to an axes instance or set of axes instances.

Once we have created an axes, we can use the ``ax.plot`` function to plot some data. Let's start with a simple sine wave:


``` python
fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
ax.plot(x, np.cos(x));
```


Alternatively, we can use the pylab interface and let the figure and axes be created for us in the background.
(See the sidebar on page X for a discussion of these two interfaces)


``` python
plt.plot(x, np.sin(x));
```


If we want to create a single figure with multiple lines, we can simply call the ``plot`` function multiple times:


``` python
plt.plot(x, np.sin(x*2));
plt.plot(x, np.sin(x));
```


That's all there is to plotting simple functions in matplotlib!
Below we'll dive into some more details about how to control the appearance of the axes and lines.

### Adjusting the Plot: Line Colors and Styles


The first adjustment you might wish to make to a plot is to control the line colors and styles.
The ``plt.plot()`` function takes additional arguments which can be used to specify these.
To adjust the color, you can use the ``color`` keyword, which accepts a string argument representing virtually any imaginable color.
The color can be specified in a variety of ways, which we'll show below:


<pre data-executable="ipython" data-code-language="python">
%pylab inline

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.plot(x, np.sin(x - 0), color='blue')        # specify color by name
plt.plot(x, np.sin(x - 1), color='g')           # short color code (works for rgb & cmyk)
plt.plot(x, np.sin(x - 2), color='0.75')        # Greyscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44')     # Hex color code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB tuple, between 0 and 1
plt.plot(x, np.sin(x - 5), color='chartreuse')  # all html color names are supported;
</pre>


Note that if no color is specified, matplotib will automatically cycle through a set of default colors for the lines.

Similarly, the line style can be adjusted using the ``linestyle`` keyword:


``` python
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted');

# For short, you can use the following codes:
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':')  # dotted;
```


If you would like to be extremely terse, these linestyle codes and color codes can be combined into a single non-keyword argument to the ``plt.plot()`` function:


``` python
plt.plot(x, x + 0, '-g')  # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r')  # dotted red;
```


These single-character color codes reflect the standard abbreviations in the RGB (Red/Green/Blue) and CMYK (Cyan/Magenta/Yellow/blacK) color systems, commonly used for digital color graphics.

There are many other keyword arguments that can be used to fine-tune the appearance of the plot: for more details, refer to matplotib's online documentation, or the docstring of the ``plt.plot()`` function.

### Adjusting the Plot: Axes limits


Matplotlib does a fairly good job of choosing default axes limits for your plot, but sometimes it's nice to have finer control.
Here we'll briefly see how to change the limits of the x and y axes.
The most basic way to do this is to use the ``plt.xlim()`` and ``plt.ylim()`` methods to set the numerical limits of the x and y axes:


``` python
plt.plot(x, np.sin(x))

plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5);
```


If for some reason you'd like either axis to be displayed in reverse, you can simply reverse the order of the arguments:


``` python
plt.plot(x, np.sin(x))

plt.xlim(10, 0)
plt.ylim(1.2, -1.2);
```


A useful related method is ``plt.axis()``: note here the potential confusion between *axes* with an *e*, and *axis* with an *i*.
This method allows you to set the x and y limits with a single call, by passing a list which specifies ``[xmin, xmax, ymin, ymax]``:


``` python
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5]);
```


The ``plt.axis()`` method goes even beyond this, allowing you to do things like automatically tighten the bounds around the current plot:


``` python
plt.plot(x, np.sin(x))
plt.axis('tight');
```


It allows even higher-level specifications, such as ensuring an equal aspect ratio so that one unit in x is equal to one unit in y:


``` python
plt.plot(x, np.sin(x))
plt.axis('equal');
```


For more information on axis limits and the other capabilities of the ``plt.axis`` method, refer to matplotlib's online documentation.

### Labeling Plots


As the last piece of this recipe, we'll briefly look at the labeling of plots: titles, axis labels, and simple legends.

Titles and axis labels are the simplest of these: there are methods which can be used to quickly set these:


``` python
plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)");
```


The position, size, and style of these labels can be adjusted using optional arguments to the function.
For more information, see the matplotlib documentation and the docstrings of each of these functions.

When multiple lines are being shown within a single axes, it can be useful to create a plot legend that labels each line type.
Again, matplotlib has a built-in way of quickly creating such a legend.
It is done via the (you guessed it) ``plt.legend()`` method.
Though there are several valid ways of using this, I find it easiest to specify the label of each line using the ``label`` keyword of the plot function:


``` python
plt.plot(x, np.sin(x), '-r', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')

plt.legend();
```


As you can see, the ``plt.legend()`` function keeps track of the line style and color, and matches these with the correct label.
More information on specifying and formatting plot legends can be found in the ``plt.legend`` doc string; additionally, we will cover some more advanced legend options in recipe X.X.

### Sidebar: Gotchas


While most ``plt`` functions translate directly to ``ax`` methods (such as ``plt.plot()`` & ``ax.plot()``), this is not always the case. In particular, functions to set limits, labels, and titles are slightly modified.
For transitioning between matlab-style functions and object-oriented methods, make the following changes:

- ``plt.xlabel()`` & ``plt.ylabel()`` become ``ax.set_xlabel()`` & ``ax.set_ylabel()``
- ``plt.xlim()`` & ``plt.ylim()`` become ``ax.set_xlim()`` & ``ax.set_ylim()``
- ``plt.title()`` becomes ``ax.set_title()``



``` python
# HIDDEN
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```


## Simple Scatter Plots


*This recipe shows a couple ways to produce simple two-dimensional scatter plots*

- ``plt.plot``
- ``plt.scatter``
- ``plt.colorbar``

In the previous recipe we saw how to create simple two-dimensional line plots with matplotlib. Here we'll look at the close cousin to these: scatter plots.

### Scatter Plots with ``plt.plot``


In the previous recipe we looked at ``plt.plot``/``ax.plot`` to produce line plots. It turns out that this function is something of a workhorse: it can produce scatter plots as well:


``` python
x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.plot(x, y, 'o', color='black');
```


The third argument in the function call is a character which represents the type of symbol used for the plotting. Just as you can specify ``'-'``, ``'--'``, etc. to control the line style, the marker style has its own set of short string codes. The full list of available symbols can be seen in the documentation of ``plt.plot``, or on the matplotlib website. Most of the possibilities are fairly intuitive, and we'll show a number of the more common ones here:


``` python
rng = np.random.RandomState(0)
for marker in ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'p', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1);
```


Using ``plt.plot``, you can also plot lines and markers and use some of the optional arguments to specify colors, sizes, etc. An example of this is below:


``` python
plt.plot(x, y, '-p', color='gray',
        markersize=15,
        markerfacecolor='white',
        markeredgecolor='gray',
        markeredgewidth=2,
        linewidth=4)
plt.ylim(-1.2, 1.2);
```


This type of flexibility in the ``plt.plot`` function allows for a wide variety of possible visualization options.
For a full description of the options available, refer to the ``plt.plot`` documentation.

### Scatter Plots with ``plt.scatter``


A second, more powerful method of creating scatterplots is the ``plt.scatter`` function, which can be used very similarly to the ``plt.plot`` function:


``` python
plt.scatter(x, y, marker='o');
```


The primary difference with ``plt.scatter`` over ``plt.plot`` is that it can be used to create scatter plots where each point has different sizes, colors, and other properties.

Let's show this by creating a random scatter plot with points of many colors and sizes.
In order to better see the overlapping results, we'll also use the ``alpha`` keyword to adjust the transparency level:


``` python
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3);
```


Adjusting the size and color of different points can offer a very useful means of visualizing multi-dimensional data. For example, we might use the iris data from scikit-learn, where each sample is one of three types of flowers which has had the size of its petals and sepals carefully measured:


``` python
from sklearn.datasets import load_iris
data = load_iris()
features = data.data.T
plt.scatter(features[0], features[1], alpha=0.2,
            s=100*features[3], c=data.target)
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.colorbar(ticks=[0, 1, 2]);
```


We can see that this scatter plot has given us the ability to simultaneously explore four different dimensions of the data set!
The (x, y) location of each point corresponds to the sepal length and width; the size of the point is related to the petal width, and the color is related to the particular species of flower.
Multi-color and multi-feature scatter-plots like this can be extremely useful for both exploration and presentation of data.

We've also introduced the function ``plt.colorbar``, which automatically places a calibrated and labeled color scaling along the side of the axes. We'll encounter this extremely useful function in many places through this chapter and the rest of the book.

### A Note on Efficiency


Aside from the different features available in ``plt.plot`` and ``plt.scatter``, why might you choose to use one over the other? While it doesn't matter as much for small amounts of data, as datasets get larger than a few thousand points, ``plt.plot`` can be noticeably more efficient than ``plt.scatter``.

The reason is simple: ``plt.scatter`` has the capability to render a different size and/or color for each point. For this reason, it internally does the extra work of constructing each point individually, even when it does not have to.
In ``plt.plot``, on the other hand, the points are always essentially clones of each other, so the work of determining the appearance of the points is done only once for the entire set of data.
When working with larger datasets, because of this implementation detail, ``plt.plot`` should be preferred whenever possible.



``` python
#HIDDEN
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```


## Visualizing Errors


*This recipe covers several methods for visualizing the errors inherent in data*

- ``plt.errorbar``
- ``plt.fill_between``

For any scientific measurement, accurate accounting for errors is nearly as important, if not more important, than accurate reporting of the number itself.
For example, imagine that I am using some astrophysical observations to estimate the Hubble Constant, the local measurement of the expansion rate of the Universe.
I know that the current literature suggests a value of around 71 (km/s)/Mpc, and I measure a value of 74 (km/s)/Mpc with my method. Are the values consistent? The only correct answer, given this information, is this: there is no way to know.

Suppose I augment this information with reported errorbars: The current literature suggests a value of around 71 $\pm$ 2.5 (km/s)/Mpc, and my method has measured a value of 74 $\pm$ 5 (km/s)/Mpc. Now are the values consistent? Most scientists would say yes.

In visualization of data and results, showing these errors well can make a plot convey much more information.

### Basic Errorbars


A basic errorbar can be created with a single matplotlib function call. We'll see this here:


``` python
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='.k');
```


Here the ``fmt`` is a format code which has the same syntax as that of ``plt.plot``, outlined in recipe X.X.
The ``errorbar`` function has many options to fine-tune the outputs.
Using these you can easily customize the aesthetics of your errorbar plot.
I often find it helpful, especially in crowded plots, to make the errorbars lighter than the points themselves:


``` python
plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0);
```


In addition to these options, you can also specify horizontal error bars (``xerr``), one-sided errorbars, and many other variants.
For more intormation on the options available, see the documentation string of ``plt.errorbar``.

### Continuous Errors


In some situations it is desirable to show errorbars on continuous quantities.
Though matplotlib does not have a built-in convenience routine for this type of application, it's relatively easy to combine primitives like ``plt.plot`` and ``plt.fill_between`` for a useful result.

Here we'll perform a simple *Gaussian Process (GP) Regression*, explored more fully in recipe X.X.
This is a method of fitting a very flexible non-parametric function to data with a continuous measure of the uncertainty.
For the meantime, we won't say anything more about Gaussian Process Regression, but will focus instead on how one might visualize such a continuous error measurement:


``` python
from sklearn.gaussian_process import GaussianProcess

# define the model and draw some data
model = lambda x: x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8])
ydata = model(xdata)

# Compute the Gaussian process fit
gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1,
                     random_start=100)
gp.fit(xdata[:, None], ydata)

xfit = np.linspace(0, 10, 1000)
yfit, MSE = gp.predict(xfit[:, None], eval_MSE=True)
dyfit = 2 * np.sqrt(MSE)  # 2*sigma ~ 95% confidence region
```


We now have ``xfit``, ``yfit``, and ``dyfit``, which sample the continuous fit to our data.
We could pass these to the ``plt.errorbar`` function as above, but we don't really want to plot 1000 points with 1000 errorbars.
Instead, we can use the ``plt.fill_between`` function with a light color to visualize this continuous error:


``` python
# Visualize the result
plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')

plt.fill_between(xfit, yfit-dyfit, yfit+dyfit,
                 color='gray', alpha=0.2)
plt.xlim(0, 10);
```


Note what we've done here with the ``fill_between`` function: we pass an x value, then the lower y-bound, then the upper y-bound, and the result is that the area between these regions is filled.

The resulting figure gives a very intuitive view into what the Gaussian Process Regression algorithm is doing: in regions near a measured data point, the model is strongly constrained and this is reflected in the small model errors.
In regions far from a measured data point, the model is not strongly constrained, and the model errors increase.

For more information on the options available in ``plt.fill_between()`` (and the closely-related ``plt.fill()`` function), see the function docstring or the matplotlib documentation.

TODO: mention/quick demo seaborn?



``` python
# HIDDEN
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```


## Density and Contour Plots


*This recipe covers several ways of visualizing two-dimensional data, such as contour and density plots*

- ``plt.contour``
- ``plt.contourf``
- ``plt.imshow``

Sometimes it is useful to display three-dimensional data in two-dimensions.
There are three matplotlib functions which can be helpful for this task: ``plt.contour`` for contour plots, ``plt.contourf`` for filled contour plots, and ``plt.imshow`` for showing images.
We'll see several examples of using these below.

### Visualizing a 3D function


We'll start by visualizing a function $z = f(x, y)$, using the following function (we've seen this before in recipe X.X, when we used it as a motivating example for broadcasting):


``` python
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
```


The ``plt.contour`` function takes three arguments: a grid of x values, a grid of y values, and a grid of z values.
The x and y values represent positions on the plot, and the z values will be represented by the contour levels.
Perhaps the most straightforward way to prepare such data is to use the ``plt.meshgrid`` function, which we explored in recipe X.X:


``` python
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
```


Now let's look at this with a standard line-only contour plot:


``` python
plt.contour(X, Y, Z, colors='black');
```


Notice that by default when a single color is used, negative values are represented by dashed lines, and positive values by solid lines.
Alternatively, the lines can be color-coded by specifying a colormap with the ``cmap`` argument.
Here, we'll also specify that we want more lines to be drawn: 20 equally-spaced intervals within the data range:


``` python
plt.contour(X, Y, Z, 20, cmap=plt.cm.cubehelix);
```


The spaces between the lines here are not ideal: let's switch to ``plt.contourf`` (notice the ``f`` at the end), which creates filled contours using the same syntax.
Additionally, we'll add a ``plt.colorbar`` command, which automatically creates an additional axis with labeled color information for the plot:


``` python
plt.contourf(X, Y, Z, 20, cmap=plt.cm.cubehelix)
plt.colorbar();
```


The colorbar makes it clear that the bright regions are "peaks", while the dark regions are "valleys".

One potential issue with this plot is that it is a bit "splotchy". That is, the color steps are discrete rather than continuous, which is not always what is desired.
This could be remedied by setting the number of contours to a very high number, but this results in a rather inefficient plot: matplotlib must render a new polygon for each step in the level.
A better way to handle this is to use the ``plt.imshow`` function, which interprets at two-dimensional grid of data as an image.
There are a few potential gotchas in this, however:

- ``plt.imshow`` doesn't accept an X and Y grid, so you must manually specify the *extent* [xmin, xmax, ymin, ymax] of the image on the plot.
- ``plt.imshow`` by default follows the standard image array definition where the origin is in the upper-left, not in the lower-left as in most contour plots. This must be changed when showing gridded data.
- ``plt.imshow`` will automatically adjust the axis aspect ratio to match the input data; this can be changed by setting, e.g. ``plt.axis(aspect='image')`` to make x and y units match.

Here is what the result looks like:


``` python
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap=plt.cm.cubehelix)
plt.colorbar()
plt.axis(aspect='image');
```


Finally, it can sometimes be useful to combine contour plots and image plots.
Here is an example of this where we'll make the image transparent (using the ``alpha`` parameter) and add labels to the contours themselves (using the ``plt.clabel`` function).


``` python
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap=plt.cm.cubehelix, alpha=0.5)
plt.colorbar()

plt.axis(aspect='image');
```


The combination of these three functions, ``plt.contour``, ``plt.contourf``, and ``plt.imshow`` gives nearly limitless possibilities for displaying this sort of three-dimensional data.

### Sidebar: Choosing a Good Color Map


This is the first recipe in which we see *colormaps*, which are matplotlib's way of mapping numbers to colors for visualization.
The default matplotlib colormap is ``jet``

TODO: talk about color blindness, conversion to black and white, and cubehelix.



``` python
#HIDDEN
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

!cp ../data/california_cities.csv ./
```


## Histograms and Binnings


*This recipe covers methods of computing and visualizing histograms and related bin-based measures of data.*

- ``plt.hist``
- ``np.histogram``
- ``plt.hist2d``
- ``np.histogram2d``
- ``plt.hexbin``
- ``scipy.stats.binned_statistic``

When exploring various datasets, a simple histogram is often one of the most useful tools.
We saw in the previous chapter a preview of matplotlib's histogram function, which creates a basic histogram in one line:


``` python
x = np.random.randn(1000)
plt.hist(x);
```


The ``hist()`` function has many options to tune both the calculation and the display, which are well explained in the documentation.
Here's an example of a more customized histogram:


``` python
plt.hist(x, bins=30, normed=True, alpha=0.3,
         histtype='stepfilled', color='green');
```


I find the combination of ``histtype='stepfilled'`` and ``alpha`` less than one to be very useful when comparing histograms of several distributions:


``` python
x1 = np.random.normal(0, 0.5, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs);
```


If you would like to simply compute the histogram (that is, count the number of points in a given bin) and not display it, the ``np.histogram()`` function is available


``` python
counts, bin_edges = np.histogram(x, bins=[-2, -1, 0, 1, 2])
print(counts)
```


### Two-dimensional Histograms and Binnings


Just as we create histograms in one dimension by dividing the number-line into bins, we can also create histograms in two-dimensions by dividing points among two-dimensional bins.
We'll take a brief look at several ways to do this here.
We'll start by defining some data: an ``x`` and ``y`` array drawn from a multivariate Gaussian distribution:


``` python
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
```


#### ``plt.hist2d``: Two-dimensional Histogram


One straightforward way to plot a 2D histogram is to use matplotlib's ``plt.hist2d`` function:


``` python
plt.hist2d(x, y, bins=30, cmap='cubehelix_r')
cb = plt.colorbar()
cb.set_label('counts in bin')
```


The ``hist2d`` function has a number of extra options to fine-tune the plot and the binning, outlined in the function docstring.
Further, just as ``plt.hist`` a counterpart in ``np.histogram``, ``plt.hist2d`` has a counterpart in ``np.histogram2d`` which can be used as follows:


``` python
counts, xedges, yedges = np.histogram2d(x, y, bins=30)
```


For the generalization of this histogram binning in dimensions higher than 2, see the ``np.histogramdd`` function.

#### ``plt.hexbin``: Hexagonal Binnings


One natural shape to use for a tesselation across a two-dimensional space is the regular hexagon.
For this purpose, matplotlib provides the ``plt.hexbin`` routine, which automatically represents a two-dimensional dataset binned within a grid of hexagons:


``` python
plt.hexbin(x, y, gridsize=30, cmap='cubehelix_r')
cb = plt.colorbar(label='count in bin')
```


``plt.hexbin`` has a number of interesting options, including the ability to specify weights for each point, and to change the output in each bin to any numpy aggregate: mean of weights, standard deviation of weights, etc.

#### Kernel Density Estimation


Another common method of evaluating densities in multiple dimensions is *Kernel density estimation* (KDE).
This will be discussed more fully in recipy X.X, but for now we'll simply mention that KDE can be thought of as a way to "smear-out" the points in space and add-up the result to obtain a smooth function.
One extremely quick and simple KDE implementation exists in the ``scipy.stats`` package.
Here we'll give a quick example of using the KDE on this data:


``` python
from scipy.stats import gaussian_kde

# fit an array of size [Ndim, Nsamples]
kde = gaussian_kde(np.vstack([x, y]))

# evaluate on a regular grid
xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

# Plot the result as an image
plt.imshow(Z.reshape(Xgrid.shape),
           origin='lower', aspect='auto',
           extent=[-3.5, 3.5, -6, 6],
           cmap='cubehelix_r')
cb = plt.colorbar()
cb.set_label("density")
```


KDE has a smoothing length which is effectively slides the knob between detail and smoothness (one example of the ubiquitous bias/variance tradeoff).
The literature on choosing an appropriate smoothing length is vast: ``gaussian_kde`` uses a rule-of-thumb to attempt to find a nearly-optimal smoothing length for the input data.

Other KDE implementations are available within the SciPy ecosystem which have various strengths and weaknesses; see for example ``sklearn.neighbors.KernelDensity`` and ``statsmodels.nonparametric.kernel_density.KDEMultivariate``.

### Binned Statistics


TODO: show ``scipy.stats.binned_statistic`` or ``plt.hexbin`` with custom ``C`` argument? What data to use?



``` python
#HIDDEN
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
!cp ../data/california_cities.csv ./
```


## Customizing Legends


*This recipe covers several ways of customizing plot legends*

- ``plt.legend``

We've previously seen briefly the use of legends in figures.
For example:


``` python
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend();
```


But there are many ways we might want to customize such a legend.
For example, we can specify the location and turn off the frame:


``` python
ax.legend(loc='upper left', frameon=False)
fig
```


We can use the ``ncol`` command to specify the number of columns in the legend:


``` python
ax.legend(frameon=False, loc='lower center', ncol=2)
fig
```


We can use a rounded box (``fancybox``) or add a shadow, or change the padding around the text:


``` python
ax.legend(fancybox=True, shadow=True, borderpad=1)
fig
```


For more information on available legend options, see the ``legend`` documentation.

### Faking the Legend


Sometimes the legend defaults are not sufficient for the given visualization.
For example, you may be using the size of points to mark certain features of the data, and want to create a legend reflecting this.
Here is an example where we'll use the size of points to indicate populations of California cities.
We'd like a legend which specifies the scale of the sizes of the points, and we'll accomplish this by plotting some labeled data with no entries:


``` python
cities = pd.read_csv('california_cities.csv')

# Extract the data we're interested in
lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

# Scatter the points, using size & color 
plt.scatter(lon, lat,
            c=np.log10(population), cmap='Greens_r',
            s=area, linewidth=0, alpha=0.5)
plt.axis(aspect='equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

# Here we create a legend: we'll plot empty lists
# with the desired size & label
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area,
                label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1)

plt.title('California Cities: Area and Population');
```


By plotting empty lists, we create labeled plot objects which are picked-up by the legend, and now our legend tells us some useful information.
This strategy can be useful in many situations

(TODO: use BaseMap here for the state boundaries?)

### Multiple Legends


Sometimes when designing a plot you'd like to add multiple legends to the same axes.
Unfortunately, matplotlib does not make this easy: via the standard ``legend`` interface, it is only possible to create a single legend for the entire plot.
If you try to create a second legend using ``plt.legend`` or ``ax.legend``, it will simply overwrite the first one.
We can work around this by creating a new legend artist from scratch, and then using ``ax.add_artist`` to manually add the second artist to the plot:


``` python
fig, ax = plt.subplots()

lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0, 10, 1000)

for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2),
                     styles[i], color='black')
ax.axis('equal')

# specify the lines and labels of the first legend
ax.legend(lines[:2], ['line A', 'line B'])

# Create the second legend and add the artist manually.
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['line C', 'line D'], loc='lower right')
ax.add_artist(leg);
```


This is a peek into the low-level artist objects which comprise any matplotlib plot.
Indeed, if you examine the source code of ``ax.legend`` (us ``ax.legend??`` within the IPython notebook) you'll see that the function simply consists of some logic to create a suitable ``Legend`` artist, which is then saved in the ``legend_`` attribute and added to the figure at draw time.



``` python
# HIDDEN
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```


## Customizing Colorbars


*This recipe covers some ideas on how to customize and effectively use colorbars on figures*

- ``plt.colorbar``
- ``plt.clim``

We've seen previously the basic use of colorbars in matplotlib plots.
The simplest colorbar can be created with the ``plt.colorbar`` function:


``` python
x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])
plt.imshow(I, cmap='RdBu')
plt.colorbar();
```


Below we'll discuss a few ideas for customizing these colorbars and using them effectively in various situations.

### Customizing Colorbars


#### Choosing the Colormap


One of the most important considerations when using colorbars is to choose an appropriate colormap, set using the ``cmap`` argument of a variety of plotting functions.
A full treatment of color choice within visualization is beyond the scope of this book, but for entertaining reading on the subject see (TODO: add reference).

Broadly, you should be aware of three different categories of colormaps:

1. *Sequential* colormaps: these are made up of one continuous sequence of colors (e.g. ``binary`` or ``cubehelix``).
2. *Divergent* colormaps: these are usually two distinct colors, which show positive and negative deviations from a mean (e.g. ``RdBu`` or ``PuOr``).
3. *Qualitative* colormaps: these mix colors with no particular sequence (e.g. ``accent`` or ``jet``).

The default matplotlib colormap, ``"jet"`` is an example of a qualitative colormap.
Its status as the default is quite unfortunate, because qualitative maps are often not a useful choice for representing quantitative data.
Among the problems is the fact that qualitative maps usually do not display any uniform progression in brightness as the scale increases.

We can see this by converting the ``jet`` colorbar into black and white:


``` python
def grayscale_cmap(cmap):
    """Return a greyscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    
    return cmap.from_list(cmap.name + "_gray", colors, cmap.N)
    

def view_colormap(cmap):
    """Plot a colormap with its greyscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))
    
    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
```



``` python
view_colormap('jet')
```


Notice the bright stripes in the result. Even in color, this sporadic brightness means that the eye will be drawn to certain portions of the color range, which will potentially emphasize unimportant parts of the dataset.
Better is to use a colormap such as Cube-Helix, which is specifically constructed to have an even brightness variation across the range. Thus it not only plays well with our color perception, but also will translate well to greyscale printing:


``` python
view_colormap('cubehelix')
```


For other situations such as showing positive and negative deviations from some mean, dual-color colorbars such as ``RdBu`` (Red-Blue) can be useful. Note, however, that the positive-negative information will be lost upon translation to greyscale!


``` python
view_colormap('RdBu')
```


We'll see an example of using some of these color maps below.

Note that there are a large number of colormaps available in matplotlib; to see a list of them you can use IPython to explore the ``plt.cm`` submodule. For a more principled approach to colors in Python, see the tools and documentation within the Seaborn library.

#### Dealing with Noise: Color Limits and Extensions


There are many available options for customization of a colorbar in matplotlib.
The colorbar itself is simply an instance of ``plt.Axes``, so all of the axes and tick formatting tricks we've learned are applicable.
The colorbar has some interesting flexibility: for example, we can narrow the color limits and indicate the out-of-bounds values with a triangular arrow at the top and bottom by setting the ``extend`` property.
This might come in handy, for example, if the image is subject to noise:


``` python
# make noise in 1% of the image pixels
speckles = (np.random.random(I.shape) < 0.01)
I[speckles] = np.random.normal(0, 3, speckles.sum())

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(I, cmap='RdBu')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')
plt.clim(-1, 1);
```


Notice that in the left panel, the default color limits respond to the noisy pixels, and the range of the noise completely washes-out the pattern we are interested in.
In the right panel, we manually set the color limits, and add extensions to indicate values which are above or below those limits.

#### Discrete Color Bars


Colormaps are by default continuous, but sometimes you'd like to represent discrete values.
There is not any easy built-in way to do this in matplotlib, but fortunately a custom discrete colormap can be quite easily constructed from any base colormap:


``` python
def discrete_cmap(N, base=None):
    """Create an N-bin discrete colormap from the specified input map"""
    cmap = plt.cm.get_cmap(base)
    color_list = cmap(np.linspace(0, 1, N))
    return cmap.from_list(cmap.name, color_list, N)
```


This discrete colormap can now be used in place of any other colormap.
For an example of where this might be useful, let's look at a manifold learning projection of scikit-learn's handwritten digits data. We'll start by downloading the digits and visualizing several of them:


``` python
# load images of the digits 1 through 6 and visualize several of them
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)

fig, ax = plt.subplots(8, 8, figsize=(6, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
```


Now let's use manifold learning to visualize the relationship between these digits.
Manifold learning is a data mining technique that we will cover more fully in recipe X.X.
These techniques are designed to project high-dimensional data into low-dimensional representations, while maintaining elements of the spatial relationships between the points.
Here each of the 64 pixels is used to define a single dimension, such that each digit can be thought of as a point in 64-dimensional space.
We'll project these 64 dimensions into just two, using the IsoMap algorithm, which preserves locality among the points:


``` python
# project the digits into 2 dimensions using IsoMap
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)
```


We'll use our discrete colormap to view the results


``` python
# plot the results
plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
            c=digits.target, cmap=discrete_cmap(6, 'cubehelix'))
plt.colorbar(ticks=range(6), label='digit value')
plt.clim(-0.5, 5.5)
```


Here we've also seen how the colorbar ticks and label can be set with simple arguments passed to the functions.
A quick note on this plot: the nonlinear mapping shows that samples of handwritten digits from 1 to 6 can be quite well separated based on their pixel values. The projection also gives us some insight: for example, 5 and 3 are much more likely to be confused by any algorithm than are 0 and 1. This observation agrees with our intuition, because 5 and 3 look much more similar than do 0 and 1.

We'll return to manifold learning and to digit recognition in a later recipe.

#### Other Colorbar Options


There are many more options available for colorbars: their size, shape, location, and orientation can be adjusted, as can the style of their ticks, labels, and color differentiations. For much more information on these, see the docstring of ``plt.colorbar`` or refer to the online matplotlib documentation.



``` python
# HIDDEN
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```


## Multiple Subplots


*This recipe covers various was to produce multiple subplots on a figure*

- ``plt.axes``
- ``plt.subplot``
- ``plt.subplots_adjust``
- ``plt.subplots``
- ``plt.GridSpec``

It is often desirable to show multiple subplots within a single figure.
These subplots might be insets, grids of plots, or other more complicated layouts.
Here we'll explore four routines that make this easy for various situations

### ``plt.axes``: subplots by-hand


The basic method of creating an axes is to use the ``plt.axes`` function.
As we've seen previously, by default this creates a standard axes object which fills the entire figure.
``plt.axes`` also takes an optional argument which is a list of four numbers in the figure coordinate system.
These numbers specify the (x, y) location on the figure (ranging from 0 to 1), and the (x, y) extent of the new axes, also ranging from 0 to 1.

For example, we might create an inset axes at the top-left of another axes by setting the x and y position to 0.65 (that is, starting at 65% of the width and 65% of the height of the figure) and the x and y extents to 0.3 (that is, the size of the axes is 20% of the width and 20% of the height of the figure):


``` python
ax1 = plt.axes()  # standard axes
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
```


The equivalent of this command within the object-oriented interface is ``fig.add_axes``. Let's use this to create two side-by-side axes:


``` python
fig = plt.figure()
ax_u = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
ax_l = fig.add_axes([0.1, 0.1, 0.8, 0.4])

x = np.linspace(0, 10)
ax_u.plot(np.sin(x))
ax_l.plot(np.cos(x));
```


We see that we now have two axes (the top with no tick labels) which are just touching: The bottom of the upper panel (at position 0.5) matches the top of the lower panel (at position 0.1 + 0.4).

### ``plt.subplot``: simple grids of subplots


Aligned columns or rows of subplots are a common-enough need that matplotlib has several convenience routines which make them easy to create.
The lowest-level of these is ``plt.subplot``, which creates a single subplot within a grid.
The ``plt.subplot`` command takes three integer arguments: the number of rows, the number of columns, and the index of the plot to be created in this scheme, which runs from the upper left to the bottom right.
For example:


``` python
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)),
             fontsize=18, ha='center')
```


The spacing between these plots can be adjusted by using the command ``plt.subplots_adjust``.
Here we'll use the equivalent object-oriented command, ``fig.add_subplot``:


``` python
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 7):
    fig.add_subplot(2, 3, i)
```


We've used the ``hspace`` and ``wspace`` arguments of ``plt.subplots_adjust``, which specify the spacing along the height and width of the figure, in units of the subplot size (in this case, the space is 40% of the subplot width and height).

## ``plt.subplots``: the whole grid in one go


Even the above can become quite tedious when creating a large grid of subplots, especially if you'd like to hide the ``x`` and ``y`` axis labels on the inner plots.
For this purpose ``plt.subplots()`` is the easier tool to use (note the ``s`` at the end of ``subplots``). Rather than creating a single subplot, this function creates a full grid of subplots in a single line, returning them in a numpy array.
The arguments are the number of rows and number of columns, along with optional keywords ``sharex`` and ``sharey``, which allow you to specify the relationships between different axes:


``` python
# Create a 2 x 3 grid, where all axes in the same row
# share a y axis, and all axes in the same column share
# an x axis
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')

# axes are in a 2D array, indexed by [row, col]
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)),
                      fontsize=18, ha='center')
```


Note that by specifying ``sharex`` and ``sharey``, we've automatically removed inner labels on the grid to make the plot cleaner.
The resulting grid of axes lies within a numpy array, allowing for convenient specification of the desired axes using standard array indexing notation.

### ``plt.GridSpec``: more complicated arrangements


Sometimes a regular grid is not what you desire. If you'd like some subplots to span multiple rows and columns, ``plt.GridSpec`` is what you're after.
The ``plt.GridSpec`` object does not create a plot by itself; it is simply a convenient interface which is recognized by the ``plt.subplot`` command.
For example, a gridspec for a grid of 2 rows and 3 columns looks like this:


``` python
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
```


We can now use slicing notation on the grid to specify plots and their extent:


``` python
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:3])
plt.subplot(grid[1, 0:2])
plt.subplot(grid[1, 2]);
```


This can be incredibly useful in many situations.
I most often use it when creating multi-axes histogram plots like the following:


``` python
# Create some normally distributed data
x, y = np.random.randn(2, 1000)

# Set up the axes with gridspec
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0, wspace=0)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# scatter points on the main axes
main_ax.plot(x, y, '.k')

# histogram on the attached axes
x_hist.hist(x, 50, histtype='stepfilled',
            orientation='vertical', color='gray')
x_hist.invert_yaxis()

y_hist.hist(y, 50, histtype='stepfilled',
            orientation='horizontal', color='gray')
y_hist.invert_xaxis()
```


Using this type of pattern with ``plt.GridSpec`` allows for virtually limitless subplot options.



``` python
# HIDDEN
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!cp ../data/memory-price.tsv ./memory-price.tsv
```


## Text and Annotation


*This recipe discusses various ways to label and annotate plots*

- ``plt.xlabel``
- ``plt.ylabel``
- ``plt.title``
- ``plt.text``
- ``plt.annotate``

Creating a good visualization is about guiding the reader so that the figure tells a story.
Sometimes, this story can be told visually, without the need for added text.
Other times, small textual cues and labels are a necessity.

Perhaps the most basic types of annotations you will use are axes labels and titles, but the options go beyond this.
Let's take a look at some data and how we might visualize and annotate it to help convey interesting information.

### The Cost of Storage over Time


For this example, we'll take a look at the cost of hard drive storage with time. This data was downloaded from http://www.mkomo.com/cost-per-gigabyte-update.
We'll start by using ``pandas`` to read-in the tab-separated data:


``` python
data = pd.read_csv('memory-price.tsv', sep='\t')
data.head()
```


We can use Pandas' simple plotting function to visualize the change in cost per GB with time:


``` python
data.plot('dateDecimal', 'dollarsPerGb', logy=True,
          linestyle='none', marker='o');
```


But let's think about how we can better represent and communicate the content of this data, using textual annotations.
Let's start by coloring the points according to the size of the storage device.
We'll also change the y-axis formatter to show that the value is dollars:


``` python
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid(True)

color = np.log10(data['sizeInGb']) // 3
color[np.isnan(color)] = -1  # missing data

ax.scatter(data['dateDecimal'], data['dollarsPerGb'],
           c=color, alpha=0.5)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('$%.2f'));
```


Now we can add an x label and a title to the plot. For axes methods, these are ``set_xlabel()``, ``set_ylabel()``, and ``set_title()``. The equivalent ``plt`` functions are ``plt.xlabel()``, ``plt.ylabel()``, and ``plt.title()``.


``` python
ax.set_xlabel('year')
ax.set_ylabel('cost (USD)')
ax.set_title('Hard Drive costs per GB')
fig
```


Finally, we'd like to give some indication of what the colors mean. We could do this usng a colorbar (see recipe X.X), but instead we might like to manually insert some text which shows the meaning.
We can do this using the ``ax.text()`` method as follows:


``` python
ax.text(1990, 1E5, 'Megabyte Drives', color='blue')
ax.text(2001, 40, 'Gigabyte Drives', color='green')
ax.text(2008, 0.03, 'Terabyte Drives', color='red', ha='right')
fig
```


The ``ax.text`` method takes an x position, a y position, a string, and then optional keywords specifying the color, size, style, and alignment of the text. Above we used ``ha='right'``, where ``ha`` is short for *horizonal alignment*.
See the docstring of ``plt.text`` for more information on available options.

#### Transforms and Text Position


Above we've anchored our text annotations to data locations. Sometimes it's preferable to anchor the text to a position on the axes or figure, independent of the data. In matplotlib, this is done by modifying the *transform*.

Any graphics display framework needs some scheme for translating between coordinate systems.
For example, a data point at $(x, y) = (1, 1)$ needs to somehow be represented at a certain location on the figure, which in turn needs to be represented in pixels on the screen.
Such coordinate transformations are mathematically relatively straightforward, and matplotlib has a well-developed set of tools that it uses internally to perform these transformations, which can be explored in the ``matplotlib.transforms`` submodule.

The average user rarely needs to worry about the details of these transforms, but it is helpful knowledge to have when considering the placement of text on a figure. There are three pre-defined transforms which can be useful in this situation:

- ``ax.transData``: transform associated with data coordinates
- ``ax.transAxes``: transform associated with the axes (in units of axes dimensions)
- ``fig.transFigure``: transform associated with the figure (in units of figure dimensions)

Here let's look at an example of drawing text at various locations using these transforms:


``` python
fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])

# transform=ax.transData is the default, but we'll specify it anyway
ax.text(1, 5, ". Data: (1, 5)", transform=ax.transData)
ax.text(0.5, 0.1, ". Axes: (0.5, 0.1)", transform=ax.transAxes)
ax.text(0.2, 0.2, ". Figure: (0.2, 0.2)", transform=fig.transFigure);
```


Note that by default, the text is aligned above and to the left of the specified coordinates: here the "." at the beginning of each string will approximately mark the given coordinate location.

The ``transData`` coordinates give the usual data coordinates associated with the x and y axis labels.
The ``transAxes`` coordinates give the location from the bottom-left corner of the axes (here the white box), as a fraction of the axes size.
The ``transFigure`` coordinates are similar, but specify the position from the bottom-left of the figure (here the gray box), as a fraction of the figure size.

Notice now that if we change the axes limits, it is only the ``transData`` coordinates that will be affected, while the others remain stationary:


``` python
ax.set_xlim(0, 2)
ax.set_ylim(-6, 6)
fig
```


This behavior can be seen more clearly by opening an interactive figure window and changing the axes limits interactively (in IPython notebook, this will require changing to an interactive backend: running the ``%matplotlib`` magic function will switch from the inline backend to the default interactive backend for your system, assuming one is available).

### Arrows and Annotation


Drawing arrows in matplotlib is often much harder than one would wish.
While there is a ``plt.arrow`` function available, I'd not suggest using it: the arrows it creates are SVG objects which will be subject to the varying aspect ratio of your plots, and the result are rarely what the user wishes.
Instead, I'd suggest using the ``plt.annotate`` function.
This function creates some text and an arrow, and the arrows can be very flexibly specified.

Below is an example of using ``annotate`` with several options


``` python
fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"));
```


There are numerous options available in the ``arrowprops`` dictionary.
Rather than enumerating all of the options, it's probably more useful to show what is possible.
Following is a nice demonstration of many available options, adapted from the official matplotlib documentation.
Notice that within arrowprops, the coordinates can be expressed in terms of data transforms or axes transforms, as we mentioned above:


``` python
# Adapted from http://matplotlib.org/examples/pylab_examples/annotation_demo2.html
from matplotlib.patches import Ellipse

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# plot a line on the first axes
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = ax1.plot(t, s, lw=3, color='purple')
ax1.axis([-1, 5, -4, 3])

# add an ellipse to the second axes
el = Ellipse((2, -1), 0.5, 0.5)
ax2.add_patch(el)
ax2.axis([-1, 5, -5, 3])

# Now for some annotations
ax1.annotate('arrowstyle', xy=(0, 1),  xycoords='data',
             xytext=(-50, 30), textcoords='offset points',
             arrowprops=dict(arrowstyle="->"))

ax1.annotate('arc3', xy=(0.5, -1),  xycoords='data',
                xytext=(-30, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=.2"))

ax1.annotate('arc #1', xy=(1., 1),  xycoords='data',
             xytext=(-40, 30), textcoords='offset points',
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="arc,angleA=0,armA=30,rad=10"))

ax1.annotate('arc #2', xy=(1.5, -1),  xycoords='data',
             xytext=(-40, -30), textcoords='offset points',
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="arc,angleA=0,armA=20,angleB=-90,armB=15,rad=7"))

ax1.annotate('angle3', xy=(2.5, -1),  xycoords='data',
             xytext=(-50, -30), textcoords='offset points',
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle3,angleA=0,angleB=-90"))

ax1.annotate('angle #1', xy=(2., 1),  xycoords='data',
             xytext=(-50, 30), textcoords='offset points',
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle,angleA=0,angleB=90,rad=10"))

ax1.annotate('angle #2', xy=(3., 1),  xycoords='data',
             xytext=(-50, 30), textcoords='offset points',
             bbox=dict(boxstyle="round", fc="0.8"),
             arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"))

ax1.annotate('angle #3', xy=(4., 1),  xycoords='data',
             xytext=(-50, 30), textcoords='offset points',
             bbox=dict(boxstyle="round", fc="0.8"),
             arrowprops=dict(arrowstyle="->",
                             shrinkA=0, shrinkB=10,
                             connectionstyle="angle,angleA=0,angleB=90,rad=10"))

ax1.annotate('angle #4', xy=(3.5, -1),  xycoords='data',
             xytext=(-70, -60), textcoords='offset points',
             size=20,
             bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle,angleA=0,angleB=-90,rad=10"))

ax1.annotate('', xy=(4., 1.),  xycoords='data',
             xytext=(4.5, -1), textcoords='data',
             arrowprops=dict(arrowstyle="<->",
                             connectionstyle="bar",
                             ec="k",
                             shrinkA=5, shrinkB=5))

ax2.annotate('$->$', xy=(2., -1),  xycoords='data',
             xytext=(-150, -140), textcoords='offset points',
             bbox=dict(boxstyle="round", fc="0.8"),
             arrowprops=dict(arrowstyle="->",
                             patchB=el,
                             connectionstyle="angle,angleA=90,angleB=0,rad=10"))

ax2.annotate('fancy', xy=(2., -1),  xycoords='data',
             xytext=(-100, 60), textcoords='offset points',
             size=20,
             arrowprops=dict(arrowstyle="fancy",
                             fc="0.6", ec="none",
                             patchB=el,
                             connectionstyle="angle3,angleA=0,angleB=-90"))

ax2.annotate('simple', xy=(2., -1),  xycoords='data',
             xytext=(100, 60), textcoords='offset points',
             size=20,
             arrowprops=dict(arrowstyle="simple",
                             fc="0.6", ec="none",
                             patchB=el,
                             connectionstyle="arc3,rad=0.3"))

ax2.annotate('wedge #1', xy=(2., -1),  xycoords='data',
             xytext=(-100, -100), textcoords='offset points',
             size=20,
             arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                             fc="0.6", ec="none",
                             patchB=el,
                             connectionstyle="arc3,rad=-0.3"))

ax2.annotate('wedge #2', xy=(2., -1),  xycoords='data',
             xytext=(35, 0), textcoords='offset points',
             size=20, va="center",
             bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),
             arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                             fc=(1.0, 0.7, 0.7), ec="none",
                             patchA=None,
                             patchB=el,
                             relpos=(0.2, 0.5)));
```


Using the above example as a reference, you should be able to construct any sort of arrow or annotation that you wish.



``` python
# HIDDEN:
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
```


## Customizing Ticks


*This recipe covers how to adjust and customize matplotlib tick locations and formats.*

- ``plt.Formatter``
- ``plt.Locator``
- ``plt.grid``

Matplotlib's default tick locators and formatters are designed to be generally sufficient in many common situations, but are in no way optimal for every plot. This recipe will give several examples of adjusting the tick locations and formatting for the particular plot type you're interested in.

Before we go into examples, it will be best for us to understand the object hierarchy of matplotlib plots. Each axes object has attributes ``xaxis`` and ``yaxis``, which contain all the properties of the lines, ticks, and labels that make up the axes.

Within each axis, there is the concept of a *major* tick mark, and a *minor* tick mark. As the names would imply, major ticks are usually bigger or more pronounced, while minor ticks are usually smaller. By default matplotlib rarely makes use of minor ticks, but one place you can see them is within logarithmic plots:


``` python
ax = plt.axes(xscale='log', yscale='log')
ax.grid() # add a grid on major ticks
```


We see here that each major tick shows by a large tickmark and a label, while each minor tick shows a smaller tickmark with no label.

These tick properties – locations and labels – can be customized by setting the ``formatter`` and ``locator`` objects of each axis. Let's examine these for the x axis of the above plot:


``` python
print(ax.xaxis.get_major_locator())
print(ax.xaxis.get_minor_locator())
```



``` python
print(ax.xaxis.get_major_formatter())
print(ax.xaxis.get_minor_formatter())
```


We see that both major and minor tick labels have their locations specified by a ``LogLocator`` (which makes sense for a logarithmic plot). Minor ticks, though, have their labels formatted by a ``NullFormatter``: this says that no labels will be shown.

Below we'll show a few examples of setting these locators and formatters for various plots.

### Removing Ticks or Labels


Sometimes for a cleaner plot, you'd like to hide ticks or labels.
This can be done using ``plt.NullLocator()`` and ``plt.NullFormatter()`` respectively.
Let's take a quick look at this:


``` python
ax = plt.axes()
ax.plot(np.random.rand(50))

ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())
```


Notice that we've removed the labels (but left the ticks) from the x axis, and removed the ticks (and thus the labels as well) from the y axis.
Having no ticks at all can be useful in many situations, for example when you want to show a grid of images. Here we'll display some of the Olivetti faces dataset, drawn from scikit-learn


``` python
fig, ax = plt.subplots(5, 5, figsize=(5, 5))
fig.subplots_adjust(hspace=0, wspace=0)

# Get some face data from scikit-learn
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces().images

for i in range(5):
    for j in range(5):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        ax[i, j].imshow(faces[10 * i + j], cmap="bone")
```


Notice that each image has its own axes, and we've set the locators to Null because the tick values (pixel number in this case) do not convey relevant information for this particular visualization.

### Reducing or Increasing the Number of Ticks


One common problem with the default settins is that smaller subplots can end up with crowded labels.
We can see this in the following plot grid:


``` python
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
```


Particularly for the x ticks, the numbers nearly overlap and make them nearly impossible to figure out.
We can fix this with the ``plt.MaxNLocator()``, which allows us to specify the maximum number of ticks which will be displayed.
Given this maximum number, matplotlib will use internal logic to choose the particular tick locations:


``` python
# For every axis, set the x & y major locator
for axx in ax.flat:
    axx.xaxis.set_major_locator(plt.MaxNLocator(3))
    axx.yaxis.set_major_locator(plt.MaxNLocator(3))
fig
```


This makes things much cleaner. If you want even more control over the locations of regularly-spaced ticks, you might also use ``plt.MultipleLocator``, as we'll see below.

### Fancy Tick Formats


Matplotlib's default tick formatting can leave a lot to be desired: it works broadly well as a default, but sometimes you'd like do do something more.
Consider the following plot, a sine and a cosine:


``` python
# Plot a sine and cosine curve
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')

# Set up grid, legend, and limits
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi);
```


There are a couple changes we might like to make. First, it's more natural for this data to space the ticks and grid lines in multiples of $\pi$. We can do this by setting a ``MultipleLocator``, which locates ticks at a multiple of the number you provide. For good measure, we'll add both major and minor ticks in multiples of $\pi/4$:


``` python
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
fig
```


But now these tick labels look a little bit silly: we can see that they are multiples of $\pi$, but the decimal representation does not immediately convey this.
To fix this, we can change the tick formatter. There's no built-in formatter for what we want to do, so we'll instead use ``plt.FuncFormatter``, which accepts a user-defined function giving fine-grained control over the tick outputs:


``` python
def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
fig
```


This is much better! Notice that we've made use of the matplotlib's LaTex support, specified by enclosing the string within dollar signs. This is very convenient for display of mathematical symbols and formulae: in this case, ``"$\pi$"`` is rendered as the Greek character $\pi$.

The ``plt.FuncFormatter()`` gives you extremely fine-grained control over the appearance of your plot ticks, and comes in very handy when preparing plots for presentation or publication.

### Summary of Formatters and Locators


We've mentioned a couple of the available formatters and locators.
We'll finish by briefly listing all the built-in locator and formatter options. For more information on any of these, refer to the docstrings or to the matplotlib online documentaion.
Each of the following is available in the ``plt`` namespace:

Locator class        | Description
---------------------|-------------
``NullLocator``      | No ticks
``FixedLocator``     | Tick locations are fixed
``IndexLocator``     | locator for index plots (e.g., where x = range(len(y)))
``LinearLocator``    | evenly spaced ticks from min to max
``LogLocator``       | logarithmically ticks from min to max
``MultipleLocator``  | ticks and range are a multiple of base
``MaxNLocator``      | finds up to a max number of ticks at nice locations
``AutoLocator``      | (default) MaxNLocator with simple defaults.
``AutoMinorLocator`` | locator for minor ticks

Formatter Class       | Description
----------------------|---------------
``NullFormatter``     | no labels on the ticks
``IndexFormatter``    | set the strings from a list of labels
``FixedFormatter``    | set the strings manually for the labels
``FuncFormatter``     | user defined function sets the labels
``FormatStrFormatter``| use a format string for each value
``ScalarFormatter``   | (default) formatter for scalar values
``LogFormatter``      | default formatter for log axes



``` python
#HIDDEN
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```


## Customizing Matplotlib: Configurations and Style Sheets


*This recipe covers tools for changing the default appearance of plots: rc parameters and stylesheets.*

Matplotlib's default plot settings are often the subject of complaint among its users.
To some degree, all plotting packages have defaults that are not optimal in some situations.
However, as some in the datavis community are fond of pointing out, matplotlib makes some objectively poor choices in some situations (particularly default color choices).
With some work and some knowledge of the available options, matplotlib can be made to produce some extremely beautiful and compelling plots.

Here we'll walk through some of matplotlib's runtime configuration (rc) options, and take a look at the relatively new *stylesheets* feature, which contains some nice sets of default configurations.

### Plot Customization By Hand


Through the recipes in this chapter, we've seen how it is possible to adjust the default plot settings to end up with something that looks a little bit nicer than the default.
It's possible to do these customizations for each individual plot.
For example, here is a fairly drab default histogram:


``` python
x = np.random.randn(1000)
plt.hist(x);
```


We can adjust this to make it a much more visually compelling plot:


``` python
# use a gray background
ax = plt.axes(axisbg='#E6E6E6')
ax.set_axisbelow(True)

# draw solid white grid lines
plt.grid(color='w', linestyle='solid')

# hide axis spines
for spine in ax.spines.values():
    spine.set_visible(False)
    
# hide top and right ticks
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# lighten ticks and labels
ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')
    
# draw semi-transparent histogram
ax.hist(x, edgecolor='#E6E6E6', color='#EE6666');
```


This looks better, and you may recognize this as inspired by the look of R's ggplot visualization package.
But this took a whole lot of effort!
We definitely don't want to have to do all that tweaking each time we create a plot.
Fortunately, there is a way to adjust these defaults once in a way that will work for all plots.

### Changing the Defaults: ``rcParams``


Each time matplotlib loads, it defines a runtime configuration (rc) which contains the default styles for every plot element you create.
This configuration can be adjusted at any time using the ``plt.rc`` convenience routine.
We won't list the full set of rc parameter options here, but these are fully enumerated in the matplotlib documentation.
Let's see what it looks like to modify the rc parameters so that our default plot will look similar to what we did above:


``` python
color_cycle = ['#EE6666', '#3388BB', '#9988DD',
               '#EECC55', '#88BB44', '#FFBBBB']
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, color_cycle=color_cycle)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)
```


With these settings defined, we can now create a plot and see our settings in action:


``` python
plt.hist(x);
```


Let's see what simple line plots look like with these rc parameters:


``` python
for i in range(4):
    plt.plot(np.random.rand(10))
```


I think this looks *much* better than the default plot type.
If you disagree with my aesthetic sense, the good news is that you can adjust the rc parameters to your own tastes!

#### Saving Your rc Settings


If you find a particular set of rc settings that you would like to use by default every time, you can create a ``matplotlibrc`` file.
Every time matplotlib is loaded, it looks for a file called ``matplotlibrc`` in the following locations:

1. The current working directory
2. A platform-specific location, (``~/.config/matplotlib/matplotlibrc`` on Linux or ``~/.matplotlib/matplotlibrc`` on OSX)
3. ``$INSTALL/matplotlib/mpl-data/matplotlibrc``, where ``$INSTALL`` is the path of your Python installation.

If a ``matplotlibrc`` file is found in any of these locations, it will be used to override the default values.

A ``matplotlibrc`` file is a simple text file which enumerates the settings you'd like to specify. For the above settings, it would look something like this:

```
# matplotlibrc file
axes.facecolor   : E6E6E6
axes.edgecolor   : none
axes.axisbelow   : True
axes.grid        : True
axes.color_cycle : EE6666, 3388BB, 9988DD, EECC55, 88BB44, FFBBBB

grid.color       : w
grid.linestyle   : solid

xtick.direction  : out
xtick.color      : gray
ytick.direction  : out
ytick.color  : gray

patch.edgecolor  : E6E6E6
lines.linewidth  : 2
```

With this saved to file in the appropriate location, the settings we used above would become the defaults whenever matplotlib is loaded.

While the matplotlibrc file may be convenient for local use, I personally prefer not to use it.
The reason is this: when you share your code, other users will probably not have the same default settings as you do.
For this reason, it's probably better to either include the rc settings within your code, or to use the stylesheet feature described below.

### Stylesheets


The 1.4 release of matplotlib in August 2014 added a very convenient ``style`` module, which includes a number of new default stylesheets, as well as the ability to create and package your own styles. These style sheets are formatted similarly to the ``matplotlibrc`` files mentioned above, but must be named with a ``.mplstyle`` extension.

Even if you don't create your own style, the stylesheets included by default are extremely useful.
You can list the available styles as follows:


``` python
plt.style.available
```


The basic way to switch to a stylesheet is to call

``` python
plt.style.use('stylename')
```

but keep in mind that this will change the style for the rest of the session!
To try out styles temporarily, you can use the style context manager:


``` python
def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))

# Default style
hist_and_lines()
```



``` python
with plt.style.context('fivethirtyeight'):
    hist_and_lines()
```



``` python
with plt.style.context('dark_background'):
    hist_and_lines()
```



``` python
with plt.style.context('ggplot'):
    hist_and_lines()
```



``` python
with plt.style.context('bmh'):
    hist_and_lines()
```



``` python
with plt.style.context('grayscale'):
    hist_and_lines()
```


With these stylesheets, you can start creating very beautiful plots without much effort!


## Statistical Visualization with Seaborn


*This recipe introduces the seaborn library, a relatively new addition to the Scientific Python ecosystem. Seaborn builds on matplotlib and provides not only beautiful plots, but a high-level interface for the creation of common statistical plot types.*

TODO: this is a place holder; I'm still not sure how deep to go.



``` python
#HIDDEN
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```


## Three-dimensional Plotting in Matplotlib


*This recipe discusses matplotlib's three-dimensional plotting capabilities.*

- ``ax.plot3D``
- ``ax.scatter3D``
- ``ax.contour3D``
- ``ax.plot_wireframe``
- ``ax.plot_surface``
- ``ax.plot_trisurf``

Matplotlib was designed to be a two-dimensional plotting library.
Around the time of the 1.0 release, some 3D plotting utilities were built on top of matplotlib's 2D display, and the result is a convenient (if rather limited) set of tools for three-dimensional data visualization.
3D plots are enabled by importing the ``mplot3d`` submodule:


``` python
from mpl_toolkits import mplot3d
```


Once this submodule is enabled, a three-dimensional axes can be created by passing the keyword ``projection='3d'`` to any of the normal axes creation routines:


``` python
fig = plt.figure()
ax = plt.axes(projection='3d')
```


With this 3D axes enabled, we can now plot a variety of three-dimensional plot types, as we'll see below. 
Three-dimensional plotting is one of the functionalities which benefits immensely from viewing figures interactively rather than statically in the notebook; recall that to use interactive figures, you can either run a stand-alone Python script with the ``plt.show()`` command, or in the IPython notebook switch to the non-inline backend using the magic command ``%matplotlib`` rather than the usual ``%matplotlib inline``.

### 3D Points and Lines


The most basic 3D plot is a line or collection of scatter plot created from sets of (x, y, z) triples.
In analogy with the more common two-dimensional discussed earlier, these can be created using the ``ax.plot3D`` and ``ax.scatter3D`` functions.
The call signature for these is nearly identical to that of their two-dimensional counterparts, so you can refer to recipe X.X for more information on controlling the output.
Here we'll plot a trigonometric spiral, along with some data drawn about the line:


``` python
ax = plt.axes(projection='3d')

# Data for a 3D line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for 3D scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
```


Notice that by default, the scatter points have their transparency adjusted to give a sense of depth on the page.
While the 3D effect is sometimes difficult to see within a static image, an interactive view can lead to some nice intuition about the layout of the points.

### 3D Contour Plots


Analogous to the contour plots we explored briefly in recipe X.X, ``mplot3d`` contains tools to create three-dimensional relief plots using the same inputs.
Like two-dimensional ``ax.contour`` plots, ``ax.contour3D`` requires all the input data to be in the form of two-dimensional regular grids, with the Z data evaluated at each point.
Here we'll show a 3D contour diagram of a three-dimenisonal sinusoidal function:


``` python
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
```



``` python
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
```


Sometimes the default viewing angle is not optimal; in this case we can use the ``view_init`` method to set the elevation and azimuthal angles. Here we'll use an elevation of 60 degrees (that is, 60 degrees above the x-y plane) and an azimuth of 35 degrees (that is, rotated 35 degrees counter-clockwise about the z-axis):


``` python
ax.view_init(60, 35)
fig
```


Again, note that this type of rotation can be accomplished interactively by clicking and dragging when using one of matplotlib's interactive backends.

### Wireframes and Surface Plots


Two other types of 3D plots which work on gridded data are wireframes and surface plots.
These take a grid of values and project it onto the specified three-dimensional surface, and can make the resulting three-dimensional forms quite easy to visualize:


``` python
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('wireframe');
```


A surface plot is like a wireframe plot, but each face of the wireframe is a filled polygon.
Adding a colormap to the filled polygons can aid perception of the topology of the surface being visualized:


``` python
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='cubehelix', edgecolor='none')
ax.set_title('surface');
```


Note that though the grid of values for a surface plot needs to be two-dimensional, it need not be rectilinear.
Here is an example of creating a partial polar grid, which when used with the ``surface3D`` plot can give us a slice into the function we're visualizing:


``` python
r = np.linspace(0, 6, 20)
theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
r, theta = np.meshgrid(r, theta)

X = r * np.sin(theta)
Y = r * np.cos(theta)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='cubehelix', edgecolor='none');
```


### Surface Triangulations


For some applications, the rectilinear grid required by the above routines is overly restrictive and inconvenient.
In these situations, the triangulation-based plots can be very useful.
What if rather than an even draw from a cartesian or a polar grid, we instead have a set of random draws?


``` python
theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)
```


We could create a scatterplot of the points to get an idea of the surface we're sampling from:


``` python
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='cubehelix', linewidth=0.5);
```


This leaves a lot to be desired.
The function that will help us in this case is ``ax.plot_trisurf``, which creates a surface by first finding a set of triangles formed between adjacent points.
Remember that ``x``, ``y``, and ``z`` here are one-dimensional arrays.


``` python
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z,
                cmap='cubehelix', edgecolor='none');
```


The result is certainly not as clean as when it is plotted with a grid, but the flexibility of such a triangulation allows for some really interesting 3D plots.
For example, it is actually possible to plot a 3D Mobius strip using this, as we'll see next.

#### Triangulation Example: Creating A Mobius Strip


The key to creating the mobius strip is to think about it's parametrization: it's a two-dimensional strip, so we need two intrinsic dimensions. Let's call them $\theta$, which ranges from $0$ to $2\pi$ around the loop, and $w$ which ranges from -1 to 1 across the width of the strip.

Let's create this parametrization:


``` python
theta = np.linspace(0, 2 * np.pi, 30)
w = np.linspace(-0.25, 0.25, 8)
w, theta = np.meshgrid(w, theta)
```


Now from this parametrization, we must determine the $(x, y, z)$ positions of the embedded strip.

The key to creating the mobius strip is to recognize that there are two rotations happening: one is the position of the loop about its center (what we've called $\theta$), while the other is the twisting of the strip about its axis (we'll call this $\phi$). For a Mobuis strip, we must have the strip makes half a twist during a full loop, or $\Delta\pi = \Delta\theta/2$.


``` python
phi = 0.5 * theta
```


Now we use simple geometry to derive the three-dimensional embedding.
We'll define $r$, the distance of each point from the center, and use this to find the embedded $(x, y, z)$ coordinates:


``` python
# radius in x-y plane
r = 1 + w * np.cos(phi)

x = np.ravel(r * np.cos(theta))
y = np.ravel(r * np.sin(theta))
z = np.ravel(w * np.sin(phi))
```


Finally, to plot the object, we must make sure the triangulization is correct. The best way to do this is to define the triangularization *within the underlying parametrization*, and then let matplotlib project this triangulation into the 3-dimensional space of the Mobius strip.
This can be accomplished as follows:


``` python
# triangulate in the underlying parametrization
from matplotlib.tri import Triangulation
tri = Triangulation(np.ravel(w), np.ravel(theta))

ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, triangles=tri.triangles,
                cmap='binary', linewidths=0.2);

ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1);
```


With techniques like this, it is possible to create and display a wide variety of 3D objects in matplotlib.

