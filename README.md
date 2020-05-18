# JavaNMF: A Java library for Non-negative Matrix Factorization

Java library for performing non-negative matrix factorization. A non-negative matrix *X* is factorized 
into the product of two matrices *W* and *H*, such that the distance between *X* and *WH* is 
minimal. The distance can be evaluated using the euclidean distance or the generalized 
Kullback-Leibler divergence with optional regularization terms. Two types of 
gradient-descent update rules are supported.

##### Implemented distance org.dulab.javanmf.measures

- Euclidean distance || *X* &minus; *WH* ||<sup>2</sup>.

##### Implemented update rules

- Multiplicative update rules for Euclidean and Kullback-Leibler distances with regularization 
terms. Based on [D. Lee and H. Seung, Algorithms for Non-negative Matrix 
Factorization](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization).

- Fast-gradient-descent update rules for Euclidean and Kullback-Leibler distance with
regularization terms. Based on [N. Guan et al., Non-Negative Patch Alignment 
Framework](http://ieeexplore.ieee.org/document/5936739/).

##### Implemented org.dulab.javanmf.algorithms

- Alternating Least Squares (ALS) method for solving the non-negative matrix factorization (NMF). 
Based on [H Kim and H, Park, Nonnegative matrix factorization based on alternating nonnegativity constrained
least squares and active set method](https://epubs.siam.org/doi/pdf/10.1137/07069239X).

- Non-negative matrix factorization (NMF), performed by alternating updates of matrices *W* and *H* to
minimize the distance between *X* and *WH*.

- Active set method for solving non-negative least squares problem. Based on [R. Bro and S.D. Jong, A fast 
non‐negativity‐constrained least squares algorithm](
https://doi.org/10.1002/(SICI)1099-128X(199709/10)11:5%3C393::AID-CEM483%3E3.0.CO;2-L).

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
DMatrixRMaj matrixW = new DMatrixRMaj(num_points, num_components);
DMatrixRMaj matrixH = new DMatrixRMaj(num_components, num_vectors);

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

Or, perform non-negative matrix factorization using the alternating least squares method.
```java
final int num_points = matrixX.rows;
final int num_vectors = matrixX.columns;

// Randomly initialize matrices W and H
Random random = new Random(0);
DMatrixRMaj matrixW = rectangle(num_points, num_components, 0.0, 1.0, random);
DMatrixRMaj matrixH = rectangle(num_components, num_vectors, 0.0, 1.0, random);

// Perform factorization
new AlternatingLeastSquaresMatrixFactorization(1e-4, 10000).solve(matrixX, matrixW, matrixH);
```

Detailed __API documentation__ can be found [here](https://du-lab.github.io/nonnegative_matrix_factorization/).

## Contributing

Code contributions are welcome. Please, contact us if you have any questions.

## Authors

- **Aleksandr Smirnov** - *Initial work* - [https://github.com/asmirn1](https://github.com/asmirn1)

## Licence

This project is licenced under the GNU GPL v2 licence - see the [LICENSE](LICENSE) file for details.

## Versions

#### Version 0.2.3

- Set a limit on the number of iterations in NonNegativeLeastSquares algorithm

#### Version 0.2.2

- Fix the error when the passive set is empty and vector S is calculated

#### Version 0.2.1

- Adds the active set method for solving Non-Negative Least Squares problem
- Adds the alternating least squares method for solving Non-Negative Least Squares problem

#### Version 0.2.0
 
- Replaces matrix library jBlas with EJML (Efficient Java Matrix Library)