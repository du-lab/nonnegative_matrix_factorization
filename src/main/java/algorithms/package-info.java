/**
 * Provides classes for performing non-negative matrix factorization, non-negative matrix regression, and
 * non-negative singular value decomposition.
 *
 * <ul>
 * <li>Non-negative matrix factorization (NMF), performed by alternating updates of matrices <i>W</i> and <i>H</i> to
 *     minimize the distance between <i>X</i> and <i>WH</i>.</li>
 * <li>Non-negative one-matrix optimization, performed by updating matrix <i>H</i> to minimize the distance between <i>X</i> and
 *     <i>WH</i>.</li>
 * <li>Non-negative singular value decomposition (NNDSVD), used to initialize matrix <i>W</i> and <i>H</i>. Based on
 *     <a href="http://www.sciencedirect.com/science/article/pii/S0031320307004359">C. Boutsidis and E. Gallopoulos,
 *     SVD based initialization: A head start for nonnegative matrix factorization]</a>.</li>
 * </ul>
 *
 * @see algorithms.MatrixFactorization
 * @see algorithms.MatrixRegression
 * @see algorithms.SingularValueDecomposition
 * @author Du-Lab Team dulab.binf@gmail.com
 */
package algorithms;