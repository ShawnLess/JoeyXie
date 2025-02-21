---
layout: distill
title: Fundamental Math for Machine Learning
description: A learning notes of the basic math knowledge and conceptions for machine learning.
tags: Math ML
giscus_comments: true
date: 2025-1-27
featured: true
# mermaid:
#  enabled: true
#  zoomable: true
# code_diff: true
# map: true
# chart:
#  chartjs: true
#  echarts: true
#  vega_lite: true
# tikzjax: true
# typograms: true

# authors:
#  - name: Albert Einstein
#    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: IAS, Princeton


# bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Linear Regression
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Logistic Regression
  - name: Generalized Linear Models
  - name: Generative Learning


# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
# _styles: >
#  .fake-img {
#    background: #bbb;
#    border: 1px solid rgba(0, 0, 0, 0.1);
#    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#    margin-bottom: 12px;
#  }
#  .fake-img p {
#    font-family: monospace;
#    color: white;
#    text-align: left;
#    margin: 12px 0;
#    text-align: center;
#    font-size: 16px;
#  }
---

## Linear Regression

Linear Regression is an algorithm which predicts unknown value with existing data set. It models the factors and results as linear function, for example:

$$  h_\theta (x)=\theta_0 + \theta_1x1 + \theta_2x2 $$

where $x_1,x_2$ are the **factors** which affect the result $h_\theta (x)$. $x_1,x_2$ are also called **features** in deep learning. 
$\theta_0, \theta_1,\theta_2$ are the parameters or **weights** in deep learning, which are supposed to be fixed during the prediction or **inference**. 

If we let $x_0$=1, then we can write the above equation in a more general form as

$$  h_\theta (x) =\sum_{i=1}^{n}\theta_ix_i=\mathbf{\theta}^T\mathbf{x}  $$

where

$$
  \mathbf{\theta} = \begin{bmatrix} \theta_0
                                \\  \theta_1
                                \\  ...
                                \\  \theta_n
                                \\
                     \end{bmatrix}
 ,   \mathbf{x} = \begin{bmatrix}   x_0
                                \\  x_1
                                \\  ...
                                \\  x_n
                                \\
                     \end{bmatrix}
$$

Given an concrete example: we suppose the house price is highly related to 1. area, and 2. number of bedroom, how can we predict the price given its area and number of bedrooms?

| Living area ($ft^2$)  |  #bedrooms  | price (k) |
| ------------------  |  ---------- | --------  |
| 2104                |  3          | 400       |
| 1600                |  3          | 330       |
| 2400                |  3          | 369       |
| 1416                |  2          | 232       |
| 3000                |  4          | 540       |
| ...                 |  ...        | ...       |

In this example:
$x_1$ is the living area, $x_2$ is the number of bedrooms, **y** is the price of the house.

The table about is called a **training set**, which will be used to training our model (that is the value of $\theta$s).

The straight thought is to choose the hypnosis **h(x)** close to the training data **y**.

This can be expressed with **cost function**. There can be many types of cost functions, the most popular one is least-squares cost function:

$$ J(\theta) = \frac{1}{2}\sum_{i=1}^{n} (h_\theta(x^i) - y^i)^2 $$

### Gradient Descent Algorithm

Gradient Descent Algorithm is used to find the value of $\theta$ so that the $J(\theta)$ is minimized, which can be described as following:

$$ \theta_j := \theta_j - \alpha \frac{\delta J(\theta ) }{\delta \theta_j} $$

Here $\delta$ is called **learning rate**.

Expend the $J(\theta)$ we can get the update equation:

$$ \theta_j := \theta_j + \alpha \sum_{i=1}^{m} (y^i - h_\theta(x^i))x_j^i $$  

where **i** is the index of data sets, **j** is the index of **features** 

#### batch gradient descent
In the above method, we look at every example in the entire training set on every step, and is called **batch gradient descent**


repeat until converge {

  $$ \theta_0 := \theta_0 + \alpha \sum_{i=1}^{m} (y^i - h_\theta(x^i))x_0^i $$

  $$ \theta_1 := \theta_1 + \alpha \sum_{i=1}^{m} (y^i - h_\theta(x^i))x_i^i $$

  $$ ... $$

  $$ \theta_n := \theta_n + \alpha \sum_{i=1}^{m} (y^i - h_\theta(x^i))x_n^i $$

}

* Pros: update of $\theta$s point to the deepest slope, which converges more quickly (TODO: illustrate with diagram)
* Cons: need to transverse the whole data set (1..m) in each step, not efficient for large training data.

#### stochastic gradient descent

If we only update the $\theta$ with current data set's gradient error, the algorithm runs more efficiently:

repeat until meet the goal {

for i= 1...m { 

$$ \theta_0 := \theta_0 + \alpha (y^i - h_\theta(x^i))x_0^i $$

$$ \theta_1 := \theta_1 + \alpha (y^i - h_\theta(x^i))x_i^i $$

$$ ... $$

$$ \theta_n := \theta_n + \alpha (y^i - h_\theta(x^i))x_n^i $$

}

}

* Pros: Efficient,especially for large training set.
* Cons: may never converge, but should be close to the target.

#### The normal equations

For linear regression, there is a more efficient way to get the value of $\theta$, which computed directly from matrix. We omit the deduction process and only give the results here.

Let matrix **X**(m..n) represents all of the x values in the training set, and vector $\overrightarrow{y}$ (n..1)

$$
  \mathbf{X} = \begin{bmatrix}x_0^0, x_1^0, ..., x_n^0
                                \\  x_0^1, x_1^1, ..., x_n^1
                                \\  ...
                                \\  x_0^m, x_1^m, ..., x_n^m
                     \end{bmatrix}
 ,   \mathbf{Y} = \begin{bmatrix}   x_0
                                \\  x_1
                                \\  ...
                                \\  x_m
                     \end{bmatrix}
$$

Then the value of $\theta$ that minimizes $J(\theta)$ is given in closed form by the equation:

$$ \theta = (X^TX)^{-1}X^T \overrightarrow{y}$$ 

This is called the **normal equation**.

#### Probabilistic interpretation of the cost function

We defined the cost function as:


$$ J(\theta) = \frac{1}{2}\sum_{i=1}^{n} (h_\theta(x^i) - y^i)^2 $$

Why we chose this? is it optimal? This can be deducted from the likelihood.

We can model the targe value y is a distribution of x:

$$ y^{(i)} = \theta ^T x^{(i)} + \epsilon ^ {(i)} $$

Where $ \epsilon ^ {(i)} $ models the unknown factor that might affect the result. **We assume** $ \epsilon ^ {(i)} $ **Normal distribution** with mean 0 and variance \sigma.  Then each sample of $y^{(i)}$ is a **conditional probability** with normal distribution whose mean value is $\theta ^T x^{(i)} $:

$$ p(y^{(i)} | x^{(i)} ;  \theta) = \frac{1}{\sqrt{2\pi\sigma}} exp (-\frac{ (y^{(i)} - \theta ^T x^{(i)}) ^2 } {2\sigma^2}) $$

Written in matrix form: 

given **X** and $\theta$,  the conditional probability of $\overrightarrow{y}$ can be write as 
$p(\overrightarrow{y} | X; \theta ）$, where $\theta$ are fixed values.

We further assume $ \epsilon ^ {(i)} $ are independent to each other, which means the distribution of $ y ^ {(i)} $ are independent to each other. Then the **likelihood** function of y can be expressed as:

$$ L(\theta) = p(\overrightarrow{y} | X; \theta ）= \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma}} exp (-\frac{ (y^{(i)} - \theta ^T x^{(i)}) ^2 } {2\sigma^2}) $$

The logic is given  $x^{(1)},x^{(2)} ... x^{(m)}$ , the probability of  $y^{(1)},y^{(2)} ... y^{(m)}$  (which are the probability that all of the **y**s occurs at the same time with the exactly value) are 

$$ p(y^{(1)} | x^{(1)}) * p(y^{(2)} | x^{(2)}) * ... * p(y^{(m)} | x^{(m)}) $$

Given all that assumption, what might be the best values of $\theta$? in **Maximum Likelihood** theory, we should choose the $\theta$ so that the probability of observed data set (that is our training set) should be maximized. 

Then the question turns out to be finding the maximum value of $L(\theta)$. To facilitate the deduction, we can also maximum $log(L(\theta))$. If we substitute $L(\theta)$, we can get the  result that the optimal value of $\theta$ are those minimizes:

$$ \frac{1}{2}\sum_{i=1}^{n} (h_\theta(x^i) - y^i)^2 $$

which are the same as **least square** cost function.

## Logistic Regression

logistic regression only outputs 0 or 1, e.g., to decide if an email is spam or not. We define a different hypotheses for this prediction, which is called **sigmod function**

$$ h_\theta(x)=\frac{1}{1-e^{-\theta^Tx}} $$

Using the **maximum likelihood** methodology, we can get update rule for stochastic descent as:

$$ \theta_j := \theta_j + \alpha \sum_{i=1}^{m} (y^i - h_\theta(x^i))x_j^i $$  

which is the same as in linear regression except the $h_\theta(x)$ is different.

## Newton's method

Newton's method is another way to find the optimal $\theta$, which can be given as:

$$ \theta := \theta - H^{-1}\triangledown_\theta l (\theta), \triangledown_\theta l (\theta)=\begin{bmatrix} \frac {\delta l (\theta)} {\theta_1}
 \\ \frac {\delta l (\theta)} {\theta_2}
 \\ ...
 \\ \frac {\delta l (\theta)} {\theta_n}
\end{bmatrix} $$

And **H** is callded **Hessian matrix**, whose element is:

$$ H_{ij}=\frac{\delta ^2 l (\theta)}  {\theta_i \theta_j} $$


## Generalized Linear Models

Most of the linear models can be expressed in a more generalized form:

$$  p(y|\eta )= b(y)e^{\eta^TT(y)-a(\eta)} $$


|   parameters      |  linear regression          |  logistic regression                | Softmax Regression              |
|  ---------------  |  ---------------            |   ---------------                   |  ---------------                |
|  $h_\theta(x)$    |  $ \theta^Tx $              |  $ \frac{1}{1+e^{-\theta^Tx}}$      | $\frac{e^{\theta_i^Tx}} {\sum_{i=1}^{k}e^{\theta_j^Tx} }$              |
|   $\eta$          |  $\mu$                      |  $ log \frac{\varphi }{1- \varphi}$ | $ log \frac{\varphi_i }{ \varphi_k}$                                   |
|   b(y)            |  $\frac{1}{\sqrt{2\pi\sigma}} exp (-\frac{ y^2}{2})$ |     1      | 1                                                                      |
|   T(y)            |  y                          |  y                                  | a matrix mapping                                                       |
|   $a(\eta)$       |  $\frac {\eta^2} {2}$       | $log( \frac{1}{1+e^\eta})$          | $-log(\varphi_k)  $                                                    |

## Generative Learning

1. **Discriminative learning**: learn **p(y\|x)** directly from the data set {**X**, $\overrightarrow{y}$}.
2. **Generative learning**: model **p(x\|y)** and **p(y)**, and then model **p(y\|x)**.

For example, if we trying to classify the pictures between monkey and dogs:

$$ \begin{cases}
 &  y=1, \text{is a monkey}\\
 &  y=0, \text{is a dog}
\end{cases} $$

For discriminative learning, it will model the conditional distribution of y given x, and will find a straight line in the space **x**, and then classify the a new animal as either as monkey or dog.

For generative learning, it first training a model of **p(x\|y=1)**, which means the distribution of x when y=1 (monkey); and then training a model of **p(x\|y=0)**, which means the distribution of x when y=0 (dog). For new pictures or new input x, we fit it into **p(x\|y=1)** and **p(x\|y=0)**, to see which model fits best and make the decisions.

After modeling **p(x\|y)** and **p(y)**, we can then get **p(x\|y)** using **Bayes rule**:

$$ p(y|x_1,x_2 ... x_n) = \frac {p(x_1,x_2,...x_n | y) p(y)} {p(x_1, x_2, ... x_3)} =  \frac {p(x_1,x_2,...x_n | y) p(y)} {p(x_1, x_2, ... x_3|y=1)p(y=1) + p(x_1, x_2, ... x_3|y=0)p(y=0) } $$

Following is more concrete example to illustrating the difference between discriminative learning and generative learning.

For logistic regression, the **average empirical loss function** is:

$$ J(\theta)= - \frac{1}{m}\sum_{i=1}^{m}y^{(i)}log(h(x^{(i)})+(1-y^{(i)})log ((1-h(x^{(i)})) $$

where $ y^{(i)} \in \{0,1\}$, $h_\theta(x)=g(\theta^Tx), g(z)=1/(1-e^{-z}), x^{(i)}=\{x_0, x_1, ..., x_n\} $. 

for **discriminative learning**, we can update the $\theta$ with Newton's method.

$$ \frac{\delta J(\theta)}{\theta_j}= -\frac{\delta}{\theta_j}\{\frac{1}{m}\sum_{i=1}^{m}y^{(i)}log(g(\theta^T x^{(i)})+(1-y^{(i)})log ((1-g(\theta^Tx^{(i)})) \} $$

$$=-\frac{1}{m}\sum_{i=1}^{m}y^{(i)} \frac{1}{(g(\theta^T x^{(i)})}g'(\theta^T x^{(i)})x^{(i)}_j + (1-y^{(i)}) \frac{-1} {((1-g(\theta^Tx^{(i)}))} g'(\theta^T x^{(i)})x^{(i)}_j $$

Given $ g'(z) = g(z)(1 - g(z)) $, the above equation become as:

$$ = - \frac{1}{m}\sum_{i=1}^{m}y^{(i)} \frac{1}{(g(\theta^T x^{(i)})}g(\theta^T x^{(i)})(1-g(\theta^T x^{(i)}))x^{(i)}_j - (1-y^{(i)}) \frac{1} {((1-g(\theta^Tx^{(i)}))} g(\theta^T x^{(i)})(1-g(\theta^T x^{(i)}))x^{(i)}_j $$ 

$$ = - \frac{1}{m}\sum_{i=1}^{m}y^{(i)} (1-g(\theta^T x^{(i)}))x^{(i)}_j - (1-y^{(i)})  g(\theta^T x^{(i)})x^{(i)}_j $$

$$ = - \frac{1}{m}\sum_{i=1}^{m}y^{(i)}x^{(i)}_j -y^{(i)}g(\theta^T x^{(i)})x^{(i)}_j - g(\theta^T x^{(i)})x^{(i)}_j + y^{(i)}) g(\theta^T x^{(i)})x^{(i)}_j $$

$$ = \frac{1}{m}\sum_{i=1}^{m}( g(\theta^T x^{(i)}) - y^{(i)} )x^{(i)}_j  $$

Written in matrix form, where X's dimension is **m..n**:

$$ \triangledown_\theta J(\theta) = \frac{1}{m} X ^T (g(X\theta) - Y) $$

We then get the **Hessian** matrix:

$$ H_{jk} = \frac{\delta ( \frac {\delta J(\theta)}   {\theta_j} )} {\theta_k}  =  \frac{\delta ( \frac{1}{m}\sum_{i=1}^{m}( g(\theta^T x^{(i)}) - y^{(i)} )x^{(i)}_j  )} {\theta_k}  $$

$$ = \frac{1}{m}\sum_{i=1}^{m}g'(\theta^T x^{(i)})x^{(i)}_j x^{(i)}_k $$

Remember $ g'(z) = g(z)(1 - g(z)) $, we get:

$$ H_{jk} = \frac{1}{m}\sum_{i=1}^{m}g(\theta^T x^{(i)}) (1 - g(\theta^T x^{(i)}) )x^{(i)}_j x^{(i)}_k $$

If we define $ d^{(i)}=g(\theta^Tx^{(i)}) (1 -  g(\theta^Tx^{(i)}))$, and D=\{ $d^{(1)}, d^{(2)}, ... , d^{(m)} $\}

then we get:

$$
H = \begin{bmatrix}
&...  &...  &... &...\\ 
&x^{(1)}_jd^{(1)} &x^{(2)}_jd^{(2)} &x^{(...)}_jd^{(...)} &x^{(m)}_jd^{(m)}  \\
&...  &... &... &... \\
\end{bmatrix}
 
\begin{bmatrix}
&...  &x^{(1)}_k  &... \\ 
&...  &x^{(2)}_k  &... \\
&...  &x^{(...)}_k  &... \\
&...  &x^{(m)}_k  &... \\
\end{bmatrix}$$

$$ = (X \bullet D)^TX $$ 

---


