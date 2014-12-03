# Simple Scatter Plots


This recipe shows a couple ways to produce simple two-dimensional scatter plots

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


<pre data-executable="ipython" data-code-language="python">
%pylab inline

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

rng = np.random.RandomState(0)
for marker in ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'p', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1);
</pre>


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

