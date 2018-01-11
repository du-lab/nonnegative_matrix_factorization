# Non-negative Matrix Factorization

Java-implementation of non-negative matrix factorization. A non-negative matrix *X* is factorized 
into the product of two matrices *W* and *H*, such that the distance between *X* and *WH* is 
minimal.

##### Implemented distance org.dulab.nmf.measures

- Euclidean distance || *X* &minus; *WH* ||<sup>2</sup>. 

- Generalized Kullback-Leibler divergence &sum; ( *X* log ( *X* / *WH* ) &minus; *X* &plus; *WH* ).

##### Implemented update rules

- Multiplicative update rules for Euclidean and Kullback-Leibler distances with regularization 
terms. Based on [D. Lee and H. Seung, Algorithms for Non-negative Matrix 
Factorization](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization).

- Fast-gradient-descent update rules for Euclidean and Kullback-Leibler distance with
regularization terms. Based on [N. Guan et al., Non-Negative Patch Alignment 
Framework](http://ieeexplore.ieee.org/document/5936739/).

##### Implemented org.dulab.nmf.algorithms

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

API documentation can be found [here](https://du-lab.github.io/nonnegative_matrix_factorization/).

## Contributing

Code contributions are welcome. Please, contact us if you have any questions.

## Authors

- **Aleksandr Smirnov** - *Initial work* - [https://github.com/asmirn1](https://github.com/asmirn1)

## Licence

This project is licenced under the GNU GPL v2 licence - see the [LICENSE](LICENSE) file for details.