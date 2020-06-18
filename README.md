# Linear Regression in Python using numpy + polyfit

I always say that learning linear regression in Python is the best first step towards machine learning. Linear regression is simple and easy to understand even if you are relatively new to data science. So spend time on 100% understanding it! If you get a grasp on its logic, it will serve you as a great foundation for more complex machine learning concepts in the future.

In this tutorial, I’ll show you everything you’ll need to know about it: the mathematical background, different use-cases and most importantly the implementation. We will do that in Python — by using numpy (polyfit).

Note: This is a hands-on tutorial. I highly recommend doing the coding part with me! If you haven’t done so yet, you might want to go through these articles first:

The mathematical background

**Here’s a quick recap!**

For linear functions, we have this formula:

> y = a*x + b

In this equation, usually, a and b are given. E.g:

> a = 2
> b = 5

So:

> y = 2*x + 5

Knowing this, you can easily calculate all y values for given x values.

E.g.

when x is... | y is...
------------ | -------------
0 |2*0 + 5 = 5
1 | 2*1 + 5 = 7
2 | 2*2 + 5 = 9
3 | 2*3 = 5 = 11
4 | 2*4 + 5 = 13
...

If you put all the `x`–`y` value pairs on a graph, you’ll get a straight line :
![linear function example](https://drive.google.com/uc?id=1u2AP2hR2aluIQrdNAtBEbf0Fav_1Dd8B)


The relationship between x and y is linear.


Using the equation of this specific line `(y = 2 * x + 5)`, if you change `x` by `1`, `y` will always change by `2`.

And it doesn’t matter what a and b values you use, your graph will always show the same characteristics: it will always be a straight line, only its position and slope change. It also means that x and y will always be in linear relationship.

In the linear function formula:

> y = a*x + b

* The `a` variable is often called slope because – indeed – it defines the slope of the red line.
* The `b` variable is called the intercept. b is the value where the plotted line intersects the y-axis. (Or in other words, the value of `y` is `b` when `x = 0`.)
![linear function example slope intercept](https://drive.google.com/uc?id=1IhIT-_CrztXVGW7PJCjuROcv0K0u0yia)


This is all you have to know about linear functions for now...


**But why did we talk so much about them?**

**Because linear regression is nothing else but finding the exact linear function equation (that is : finding the `a` and `b` values in the `y = a*x + b` formula) that fits your data points the best.**

Note : Here’s some advice if you are not 100% sure about the math. The most intuitive way to understand the linear function formula is to play around with its values. Change the a and b variables above, calculate the new x-y value pairs and draw the new graph. Repeat this as many times as necessary. (Tip: try out what happens when `a = 0` or `b = 0`!) By seeing the changes in the value pairs and on the graph, sooner or later, everything will fall into place.

# A typical linear regression example

Machine learning – just like statistics – is all about abstractions. You want to simplify reality so you can describe it with a mathematical formula. But to do so, you have to ignore natural variance — and thus compromise on the accuracy of your model.

If this sounds too theoretical or philosophical, here’s a typical linear regression example!

We have 20 students in a class and we have data about a specific exam they have taken. Each student is represented by a blue dot on this scatter plot:

* The **X axis** shows how many hours a student studied for the exam
* The **Y axis** shows the scores that she eventually got
![linear function example](https://drive.google.com/uc?id=1ec76ma1WGX0AgRbtuJQvlysp2bx8g5SN)


E.g. she studied 24 hours and her test result was 58% :

![linear regression scatter plot example one student](https://drive.google.com/uc?id=1zcw5sQ1-j-FyX3cFc0XdVyedPyREQZz1)
We have 20 data points (20 students) here.


By looking at the whole data set, you can intuitively tell that there must be a correlation between the two factors. If one studies more, she’ll get better results on her exam. But you can see the natural variance, too. For instance, these 3 students who studied for ~30 hours got very different scores: 74%, 65% and 40%
![scatter plot spread](https://drive.google.com/uc?id=1l8Wk1PbzspDu39r1UpyifJhsCjFmJ1nb)


Anyway, let’s fit a line to our data set — using linear regression:
![linear regression fitted line](https://drive.google.com/uc?id=1xfTijZu8u1PtfQ9xOsF0k0aHDtJNVVKC)


Nice, we got a line that we can describe with a mathematical equation – this time, with a linear function. The general formula was: 

> y = a * x + b

And in this specific case, the a and b values of this line are :

> a = 2.01
> b = -3.9

So the exact equation for the line that fits this dataset is :

> y = 2.01*x - 3.9

And how did I get these `a` and `b` values? By using machine learning.

If you know enough `x`–`y` value pairs in a dataset like this one, you can use linear regression machine learning algorithms to figure out the exact mathematical equation (so the `a` and `b` values) of your linear function.

# Linear regression terminology

Before we go further, I want to talk about the terminology itself — because I see that it confuses many aspiring data scientists. Let’s fix that here!

Okay, so one last time, this was our linear function formula :

> y = a*x + b

The `a` and `b` variables :

The `a` and `b` variables in this equation define the position of your regression line and I’ve already mentioned that the a variable is called slope (because it defines the slope of your line) and the b variable is called intercept.

In the machine learning community the a variable (the slope) is also often called the regression coefficient.

The `x` and `y` variables:

The `x` variable in the equation is the input variable — and `y` is the output variable.
This is also a very intuitive naming convention. For instance, in this equation :

> y = 2.01*x - 3.9

If your input value is `x = 1`, your output value will be `y = -1.89`.

But in machine learning these x-y value pairs have many alternative names… which can cause some headaches. So here are a few common synonyms that you should know:

* **input variable (`x`) – output variable (`y`)
* **independent variable (`x`) – dependent variable (`y`)
* **predictor variable (`x`) – predicted variable (`y`)
* **feature (`x`) – target (`y`)

See, the confusion is not an accident… But at least, now you have your linear regression dictionary here.

# How does linear regression become useful ?

Having a mathematical formula – even if it doesn’t 100% perfectly fit your data set – is useful for many reasons.

1. **Predictions** : Based on your linear regression model, if a student tells you how much she studied for the exam, you can come up with a pretty good estimate: you can predict her results even before she writes the test. Let’s say someone studied 20 hours; it means that her predicted test result will be `2.01 * 20 - 3.9 = 36.3`.

2. **Outliers** : If something unexpected shows up in your dataset – someone is way too far from the expected range…

![outlier detection with linear regression](https://drive.google.com/uc?id=1XSH-mDKrxluHUZn0ih2Zb4oGrTKaChA_)


… let’s say, someone who studied only 18 hours but got almost 100% on the exam… Well, that student is either a genius — or a cheater. But she’s definitely worth the teachers’ attention, right? 🙂 By the way, in machine learning, the official name of these data points is **outliers**.

And both of these examples can be translated very easily to real life business use-cases, too!

**Predictions** are used for: sales predictions, budget estimations, in manufacturing/production, in the stock market and in many other places. (Although, usually these fields use more sophisticated models than simple linear regression.)

Finding outliers is great for fraud detection. And it’s widely used in the fintech industry. (E.g. preventing credit card fraud.)

# The limitations of machine learning models

It’s good to know that even if you find a very well-fitting model for your data set, you have to count on some limitations.

Note : These are true for essentially all machine learning algorithms — not only for linear regression.

**Limitation #1 : a model is never a perfect fit

As I said, fitting a line to a dataset is always an abstraction of reality. Describing something with a mathematical formula is sort of like reading the short summary of Romeo and Juliet. You’ll get the essence… but you will miss out on all the interesting, exciting and charming details. 

Similarly in data science, by “compressing” your data into one simple linear function comes with losing the whole complexity of the dataset: you’ll ignore natural variance.

But in many business cases, that can be a good thing. Your mathematical model will be simple enough that you can use it for your predictions and other calculations. 

Note : One big challenge of being a data scientist is to find the right balance between a too-simple and an overly complex model — so the model can be as accurate as possible. (This problem even has a name: bias-variance tradeoff, and I’ll write more about this in a later article.)

**But a machine learning model – by definition – will never be 100% accurate.

**Limitation #2 : you can’t go beyond the range of your historical data

Many data scientists try to extrapolate their models and go beyond the range of their data.

For instance, in our case study above, you had data about students studying for 0-50 hours. The dataset hasn’t featured any student who studied 60, 80 or 100 hours for the exam. These values are out of the range of your data. If you wanted to use your model to predict test results for these “extreme” `x` values... well you would get nonsensical `y` values :

E.g. your model would say that someone who has studied `x = 80` hours would get:

> y = 2.01*80 - 3.9 = 159% on the test
![extrapolate issue](https://drive.google.com/uc?id=https://drive.google.com/file/d/1dB4AIpaY6BLAQhSw-yPKAIP8iPt25E4c)


...but 100% is the obvious maximum, right?

**The point is that you can’t extrapolate your regression model beyond the scope of the data that you have used creating it. Well, in theory, at least...

Because I have to admit, that in real life data science projects, sometimes, there is no way around it. If you have data about the last 2 years of sales — and you want to predict the next month, you have to extrapolate. Even so, we always try to be very careful and don’t look too far into the future. The further you get from your historical data, the worse your model’s accuracy will be.

# Linear Regression in Python

Okay, now that you know the theory of linear regression, it’s time to learn how to get it done in Python !

Let’s see how you can fit a simple linear regression model to a data set !

Well, in fact, there is more than one way of implementing linear regression in Python. Here, I’ll present my favorite — and in my opinion the most elegant — solution. I’ll use `numpy` and its `polyfit` method.

We will go through these 6 steps:

1. Importing the Python libraries we will use
2. Getting the data
3. Defining x values (the input variable) and y values (the output variable)
4. Machine Learning: fitting the model
5. Interpreting the results (coefficient, intercept) and calculating the accuracy of the model
6. Visualization (plotting a graph)

You might ask : “Why isn’t Tomi using `sklearn` in this tutorial?” I know that (in online tutorials at least) `Numpy` and its `polyfit` method is less popular than the Scikit-learn alternative… true. But in my opinion, `numpy`‘s `polyfit` is more elegant, easier to learn — and easier to maintain in production ! `sklearn`‘s linear regression function changes all the time, so if you implement it in production and you update some of your packages, it can easily break. I don’t like that. Besides, the way it’s built and the extra data-formatting steps it requires seem somewhat strange to me. In my opinion, sklearn is highly confusing for people who are just getting started with Python machine learning algorithms. (By the way, I had the `sklearn` LinearRegression solution in this tutorial... but I removed it. That’s how much I don’t like it. So trust me, you’ll like `numpy` + `polyfit` better, too. :-))
