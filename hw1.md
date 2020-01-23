# Homework 1

### **Problem 1**

$L_{2} = \frac{1}{N}  \sum_{\forall_{n}} (w_{0} + w_{1}x_n - t_{n})^{2}$

#### Partial Div of: $w_0$

$\frac{\partial L_{2}}{\partial w_{0}} = \frac{1}{N} \sum_{\forall_{n}} (w_{0} + w_{1}x_n - t_{n})^{2}\frac{\partial L_{2}}{\partial w_{0}}$

$= \frac{1}{N} * \sum_{\forall_{n}} (w_{0} + w_{1}x_n - t_{n})^{2} \frac{\partial L_{2}}{\partial w_{0}}$

$= \frac{1}{N} *  \sum_{\forall_{n}} 2 (w_{0} + w_{1}x_n - t_{n}) * (w_{0} + w_{1}x_n - t_{n}) \frac{\partial L_{2}}{\partial w_{0}}$ ; **Chain Rule**

$= \frac{1}{N} *  \sum_{\forall_{n}} 2 (w_{0} + w_{1}x_n - t_{n}) * (w_{0} \frac{\partial L_{2}}{\partial w_{0}} + w_{1}x_n\frac{\partial L_{2}}{\partial w_{0}} - t_{n}\frac{\partial L_{2}}{\partial w_{0}})$ ; **Chain Rule**

$= \frac{1}{N} *  \sum_{\forall_{n}} 2 (w_{0} + w_{1}x_n - t_{n}) * (1+ 0 - 0)$

>$= \frac{\sum_{\forall_{n}} 2 (w_{0} + w_{1}x_n - t_{n})}{N}$

#### Solving For $w_0$

$0= \frac{ \sum_{\forall_{n}} 2 (w_{0} + w_{1}x_n - t_{n})}{N}$
$=> \sum_{\forall_{n}} (w_{0} + w_{1}x_n - t_{n})$

$0= \sum_{\forall_{n}} (w_{0} + w_{1}x_n - t_{n})$

>$\sum_{\forall_{n}} t_{n} - w_{1}x_n = w_{0}$

#### Partial Div of: $w_1$

$\frac{\partial L_{2}}{\partial w_{1}} = \frac{1}{N} \sum_{\forall_{n}} (w_{0} + w_{1}x_n - t_{n})^{2}\frac{\partial L_{2}}{\partial w_{1}}$

$= \frac{1}{N} * \sum_{\forall_{n}} (w_{0} + w_{1}x_n - t_{n})^{2} \frac{\partial L_{2}}{\partial w_{1}}$

$= \frac{1}{N} *  \sum_{\forall_{n}} 2 (w_{0} + w_{1}x_n - t_{n}) * (w_{0} + w_{1}x_n - t_{n}) \frac{\partial L_{2}}{\partial w_{1}}$ ; **Chain Rule**

$= \frac{1}{N} *  \sum_{\forall_{n}} 2 (w_{0} + w_{1}x_n - t_{n}) * (w_{0} \frac{\partial L_{2}}{\partial w_{1}} + w_{1}x_n\frac{\partial L_{2}}{\partial w_{1}} - t_{n}\frac{\partial L_{2}}{\partial w_{1}})$ ; **Chain Rule**

$= \frac{1}{N} *  \sum_{\forall_{n}} 2 (w_{0} + w_{1}x_n - t_{n}) * (0+ x_n - 0)$

>$= \frac{\sum_{\forall_{n}} 2x_n (w_{0} + w_{1}x_n - t_{n})}{N}$

#### Solving For $w_1$

$0= \frac{\sum_{\forall_{n}} 2x_n (w_{0} + w_{1}x_n - t_{n})}{N}$
$=> \sum_{\forall_{n}} (2x_n w_{0} +2x_n w_{1}x_n - 2x_n t_{n})$

$0 = \sum_{\forall_{n}} (2x_n w_{0} +2x_n w_{1}x_n - 2x_n t_{n})$

>$\sum_{\forall_{n}} \frac{t_{n} - w_0}{x_n} = w_1$


