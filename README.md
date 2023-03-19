# Ensemble-Learning Group Project

by Congjie AN, Jialin DU, Zeli PAN, Yifan WANG

This is the ensemble-learning group project of DSBA program, ESSEC & CentraleSupélec. This group project consists of two parts: a kaggle competition to predict Airbnb price in New York and an implementation of decision tree from scratch.

For each part, there are detailed steps in the jupyter notebooks. The content of this readme is similar to the report.

---

---

## PART 2 Decision Tree Implementation

In this part, we use **only numpy** to implement a Classification And Regression Tree (CART) from scratch. The entire implementation is divided into four parts:
- 1.1 Construction of the CART.
- 1.2 Evaluation on both classification and regression tasks.
- 2.1 An evolutionary algorithm for hyperparameter tuning.
- 2.2 Optimized CART performance.

### 2.1.1 CART Construction

In this part, we use a lot of recursive functions to reduce the complexity of the code by sacrificing memories. We first define a function to calculate the impurity:

![impurity](images/image.jpg)
