# Feature-Reduction-using-Genetic-Algorithms
We have many feature reduction techniques in this age of AI, but we have tried to dentify essential features (from the entire features space), those can suffice for better performance of a model using Genetic Algorithms because of their simplicity and resemblence of the natural evolution. We've tried to understand the perfromance of GA, and how they stand against existing approaches. This is an attempt, and might not be the best, that puts all my understanding of Genetic Algotihms to identify the sufficient features space that can produce better fitness score or accuracy. Dataset used is from unknown source provided by professor, so couldnot provide source of it.


### Feature Selection using Genetic Algorithms

Genetic algorithms (GAs) can effectively optimize feature selection in machine learning and data mining tasks. Feature selection aims to identify the most relevant subset of features from a dataset to improve the performance of machine learning models by reducing dimensionality and removing irrelevant or redundant features.

#### Objective
Select the optimal features from the feature vector using Genetic Algorithms.

#### Dataset Description
- The dataset consists of 2708 documents (data instances) classified into one of seven classes \( y = \{0, 1, 2, 3, 4, 5, 6\} \).
- Each document is represented by a feature vector of 1433 features with a binary (0/1) value indicating the absence/presence of the corresponding feature.

#### Classifier Description
- Let \( X_{yi} \) be the average of the vectors of all training documents in class \( y_i = \{0, 1, 2, 3, 4, 5, 6\} \).
- Let \( q \) be a data instance in the validation set.
- Cosine similarity between \( X_{yi} \) and \( q \) is calculated as:
  <p align = 'center'>
    <img src = 'https://github.com/pbt12/Feature-Reduction-using-Genetic-Algorithms/assets/74967927/9456a4f3-b234-4e0d-9332-11132e1a47e5'/>
  </p>

Class label 𝑦 is assigned to 𝑞 by:
<p align = 'center'>
    <img src = 'https://github.com/pbt12/Feature-Reduction-using-Genetic-Algorithms/assets/74967927/13a8b239-0eac-47e2-b63a-0c0a267e4ee2'/>
  </p>

  System accuracy is measured using the following formula:
<p align = 'center'>
    <img src = 'https://github.com/pbt12/Feature-Reduction-using-Genetic-Algorithms/assets/74967927/476c34bf-b1c3-4bc2-871c-8ecc8bdaabd3'/>
  </p>

- $ \( \hat{y}_i \) is the assigned class label by the system,
\( y_i \) is the true class label of the document \( x_i \) in \( D_{validate} \),
and \( T \) is the number of documents in \( D_{validate} \). $

#### Desing and Implementation
Here's a quick breakdown of a typical Generic Algorithm:
- **Population of Solutions**: Imagine a bunch of candidate solutions to your problem. This is the population, and each solution is like an individual chromosome
- **Selection**: Not all solutions are created equal. The GA selects the better solutions (based on a fitness function) to be parents for the next generation
- **Crossover**: Parents swap genetic information (like mixing DNA in reproduction) to create children (new solutions) that inherit traits from both parents
- **Mutation**:  Small random changes are introduced to the children's solutions, mimicking mutations that can happen in nature. This helps maintain diversity and avoid getting stuck in local optima
- **Repeat**: This cycle (selection, crossover, mutation) is repeated over generations, with the population evolving towards better and better solutions


The Genetic algorithm using a selection, crossover and mutation criterion progresses towards imcreasing the fitness_score or accuracy of the classifier with increasing generations.

``` GA.py ``` is our initial draft of code for genetic algorithm, and ``` Genetic Algorithms.ipynb ``` is the ipynb file that contains all our experimental details and results.
