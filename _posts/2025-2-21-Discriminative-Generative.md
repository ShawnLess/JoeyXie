---
layout: post
title: Discriminative learning vs Generative learning
description: A concrete comparison of the discriminative and generative learning.
tags: Math ML
giscus_comments: true
date: 2025-2-21
featured: false
---

Taking logistic regression as an example, the **average empirical loss function** is:

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


