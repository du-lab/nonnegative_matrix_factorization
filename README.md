# JavaNMF: A Java library for Non-negative Matrix Factorization

Java library for performing non-negative matrix factorization. A non-negative matrix *X* is factorized 
into the product of two matrices *W* and *H*, such that the distance between *X* and *WH* is 
minimal. The distance can be evaluated using the euclidean distance or the generalized 
Kullback-Leibler divergence with optional regularization terms. Two types of 
gradient-descent update rules are supported.

##### Implemented distance org.dulab.javanmf.measures

- Euclidean distance || *X* &minus; *WH* ||<sup>2</sup>. 

- Generalized Kullback-Leibler divergence &sum; ( *X* log ( *X* / *WH* ) &minus; *X* &plus; *WH* ).

##### Implemented update rules

- Multiplicative update rules for Euclidean and Kullback-Leibler distances with regularization 
terms. Based on [D. Lee and H. Seung, Algorithms for Non-negative Matrix 
Factorization](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization).

- Fast-gradient-descent update rules for Euclidean and Kullback-Leibler distance with
regularization terms. Based on [N. Guan et al., Non-Negative Patch Alignment 
Framework](http://ieeexplore.ieee.org/document/5936739/).

##### Implemented org.dulab.javanmf.algorithms

- Non-negative matrix factorization (NMF), performed by alternating updates of matrices *W* and *H* to
minimize the distance between *X* and *WH*.

- Non-negative optimization, performed by updating matrix *H* to minimize the distance between 
*X* and *WH*.

- Non-negative singular value decomposition (NNDSVD), used to initialize matrix *W* and *H*. Based on 
[C. Boutsidis and E. Gallopoulos, SVD based initialization: A head start for nonnegative matrix 
factorization](http://www.sciencedirect.com/science/article/pii/S0031320307004359).

##  Getting Started

These instructions will get you a copy of the project up and running on your local machine for 
development and testing purposes.

### Prerequisites

- [Java 1.8](https://java.com/en/download/) or newer.

- [Maven 3](https://maven.apache.org/)  or newer.

### Installation

To install the package, clone this project and use maven to build it:
```
git clone https://github.com/du-lab/nonnegative_matrix_factorization.git
cd nonnegative_matrix_factorization/
mvn clean install
``` 

## Documentation

__Example__: Given `matrixX` and `num_components`, perform non-negative matrix factorization using the euclidean distance with regularization, multiplicative update rule,
and NNDSVD-initialization.

```java
final int num_points = matrixX.rows;
final int num_vectors = matrixX.columns;

// Create matrices W and H
DoubleMatrix matrixW = new DoubleMatrix(num_points, num_components);
DoubleMatrix matrixH = new DoubleMatrix(num_components, num_vectors);

// Initialize matrices W and H by the NNDSVD-method
new SingularValueDecomposition(matrixX).decompose(matrixW, matrixH);

// Choose update rules for matrices W and H:
// Multiplicative update rule for the euclidean distance with l1-regularization
UpdateRule updateRuleW = new MUpdateRule(1.0, 0.0);
// Multiplicative update rule for the euclidean distance with l2-regularization
UpdateRule updateRuleH = new MUpdateRule(0.0, 1.0);

// Perform factorization
new MatrixFactorization(updateRuleW, updateRuleH, 1e-4, 10000).execute(matrixX, matrixW, matrixH);
```

Detailed __API documentation__ can be found [here](https://du-lab.github.io/nonnegative_matrix_factorization/).

## Contributing

Code contributions are welcome. Please, contact us if you have any questions.

## Authors

- **Aleksandr Smirnov** - *Initial work* - [https://github.com/asmirn1](https://github.com/asmirn1)

## Licence

This project is licenced under the GNU GPL v2 licence - see the [LICENSE](LICENSE) file for details.

## Versions

#### Version 0.2.1

- Adds the active set method for solving Non-Negative Least Squares problem

#### Version 0.2.0
 
- Replaces matrix library jBlas with EJML (Efficient Java Matrix Library)