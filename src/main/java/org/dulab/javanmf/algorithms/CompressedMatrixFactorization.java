/*
 * Copyright (C) 2018 Du-Lab Team <dulab.binf@gmail.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

package org.dulab.javanmf.algorithms;

import org.dulab.javanmf.measures.Measure;
import org.dulab.javanmf.updaterules.UpdateRule;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.interfaces.decomposition.QRDecomposition;

import javax.annotation.Nonnull;
import java.util.Random;
import java.util.logging.Logger;

import static org.ejml.dense.row.CommonOps_DDRM.*;

/**
 * This class performs compressed non-negative matrix factorization: for given matrix X, find matrices W and H that minimize the
 * objective function
 * <p>
 * &emsp; D(X, WH) + &lambda;<sub>w</sub>||W||<sub>1</sub> + &lambda;<sub>h</sub>||H||<sub>1</sub> +
 * 0.5 &mu;<sub>w</sub>||W||<sup>2</sup> + 0.5 &mu;<sub>h</sub>||H||<sup>2</sup>
 * <p>
 * where D(X, WH) is either the euclidean distance or the Kullback-Leibler divergence, ||&middot;|| is the
 * Frobenius norm, and ||&middot;||<sub>1</sub> is the <i>l</i><sub>1</sub>-norm.
 *
 * See M. Tepper and G. Sapiro. "Compressed Nonnegative Matrix Factorization
 * Is Fast and Accurate" for details.
 *
 * <p>
 * <strong>Example</strong> for given matrix {@code matrixX}, matrices {@code matrixW} and {@code matrixH} are modified
 * to minimize the euclidean distance with regularization.
 * <pre> {@code
 *     UpdateRule updateRuleW = new MUpdateRule(1.0, 0.0);
 *     UpdateRule updateRuleH = new MUpdateRule(0.0, 1.0);
 *
 *     MatrixFactorization factorization = new MatrixFactorization(
 *         updateRuleW, updateRuleH, 1e-6, 10000);
 *
 *     factorization.execute(matrixX, matrixW, matrixH);
 * } </pre>
 *
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public class CompressedMatrixFactorization {
    /* Logger */
    private static final Logger LOG = Logger.getLogger(CompressedMatrixFactorization.class.getSimpleName());

    /**
     * Tolerance of the fitting error
     */
    private final double tolerance;

    /**
     * Maximum number of iterations
     */
    private final int maxIteration;

    private final Random rand;

    /* Update rule for matrix W */
    private final UpdateRule updateRuleW;

    /* Update rule for matrix H */
    private final UpdateRule updateRuleH;

    /* Distance measure associated with the update rules */
    private final Measure measure;

    /**
     * Creates an instance of {@link CompressedMatrixFactorization}
     *
     * @param updateRuleW  instance of {@link UpdateRule} for matrix W
     * @param updateRuleH  instance of {@link UpdateRule} for matrix H
     * @param tolerance    the fitting error tolerance
     * @param maxIteration maximum number of iterations to use
     */
    public CompressedMatrixFactorization(@Nonnull UpdateRule updateRuleW, @Nonnull UpdateRule updateRuleH,
                                         double tolerance, int maxIteration) {
        this.updateRuleW = updateRuleW;
        this.updateRuleH = updateRuleH;
        this.measure = updateRuleW.measure;
        this.tolerance = tolerance;
        this.maxIteration = maxIteration;
        this.rand = new Random(0);
    }

    /**
     * Performs the non-negative matrix factorization with given initial matrices W and H.
     * <p>
     * Parameters {@code w} and {@code h} contain the result of the factorization.
     *
     * @param data    matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>], a collection of vectors in N<sub>points</sub>-dimensional space
     * @param w       matrix of shape [N<sub>points</sub>, N<sub>components</sub>], a collection of initial components
     * @param h       matrix of shape [N<sub>components</sub>, N<sub>vectors</sub>], a collection of initial coefficients
     * @param verbose flag to output verbose information
     */
    public void execute(@Nonnull DMatrixRMaj data, @Nonnull DMatrixRMaj w, @Nonnull DMatrixRMaj h, boolean verbose) {

        DMatrixRMaj x = new DMatrixRMaj(data);
        DMatrixRMaj xt = transpose(x, null);

        DMatrixRMaj matrixL = compressMatrix(x, w.numCols);
        DMatrixRMaj matrixR = transpose(compressMatrix(xt, w.numCols), null);

        DMatrixRMaj matrixXHat = new DMatrixRMaj(matrixL.numCols, x.numCols);
        multTransA(matrixL, x, matrixXHat);

        DMatrixRMaj matrixXCheck = new DMatrixRMaj(x.numRows, matrixR.numRows);
        multTransB(x, matrixR, matrixXCheck);
        DMatrixRMaj matrixXCheckT = transpose(matrixXCheck, null);

        DMatrixRMaj wt = transpose(w, null);
        DMatrixRMaj wttBuffer = new DMatrixRMaj(w.numRows, w.numCols);

        final double initError = measure.get(x, w, h);
        double prevError = initError;

//        DMatrixRMaj htBuffer = new DMatrixRMaj(h.numCols, h.numRows);
//        DMatrixRMaj wttBuffer = new DMatrixRMaj(w.numRows, w.numCols);
        DMatrixRMaj rhtBuffer = new DMatrixRMaj(matrixR.numRows, h.numRows);
        DMatrixRMaj ltwBuffer = new DMatrixRMaj(matrixL.numCols, w.numCols);

        // Update matrices WT and H until the error is small or the maximum number of iterations is reached
        int k;
        for (k = 1; k < maxIteration + 1; ++k) {

            multTransB(matrixR, h, rhtBuffer);
            updateRuleW.update(matrixXCheckT, rhtBuffer, wt);

            multTransAB(matrixL, wt, ltwBuffer);
            updateRuleH.update(matrixXHat, ltwBuffer, h);

//            updateRuleH.update(x, transpose(wt, wttBuffer), h);
//            updateRuleW.update(xt, transpose(h, htBuffer), wt);

            if (k % 10 == 0) {
                double error = measure.get(x, transpose(wt, wttBuffer), h);
                if (Math.abs(prevError - error) / initError < tolerance) {
                    if (verbose) LOG.info("NMF is completed after " + k + " iterations");
                    break;
                }
                prevError = error;
            }
        }

        if (verbose && k >= maxIteration)
            LOG.info("NMF does not converge after " + k + " iterations");

        transpose(wt, w);
//        w.copy(wt.transpose());
    }

    /**
     * Performs the non-negative matrix factorization with given initial matrices W and H.
     * <p>
     * Parameters {@code w} and {@code h} contain the result of the factorization.
     *
     * @param data matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>], a collection of vectors in
     *             N<sub>points</sub>-dimensional space
     * @param w    matrix of shape [N<sub>points</sub>, N<sub>components</sub>], a collection of initial components
     * @param h    matrix of shape [N<sub>components</sub>, N<sub>vectors</sub>], a collection of initial coefficients
     */
    public void execute(@Nonnull DMatrixRMaj data, @Nonnull DMatrixRMaj w, @Nonnull DMatrixRMaj h) {
        execute(data, w, h, false);
    }

    /**
     * Creates a compressed matrix at described in Table 1 of the paper
     * @param a matrix to be compressed
     * @param rank rank of matrix (number of components)
     * @return compressed matrix
     */
    private DMatrixRMaj compressMatrix(DMatrixRMaj a, int rank) {
        int exponent = 4;
        int numCols = Math.min(Math.max(20, rank + 10), a.numCols);
        return compressMatrix(a, numCols, exponent);
    }

    /**
     * Creates a compressed matrix at described in Table 1 of the paper
     * @param a matrix to be compressed
     * @param numCols number of columns in the compressed matrix
     * @param exponent parameter w (see Table 1 in the paper)
     * @return compressed matrix
     */
    private DMatrixRMaj compressMatrix(DMatrixRMaj a, int numCols, int exponent) {

        DMatrixRMaj omega = RandomMatrices_DDRM.rectangleGaussian(
                a.numCols, numCols, 0.0, 1.0, rand);
        abs(omega);

        // Calculate (A*AT)^exponent
        DMatrixRMaj product1 = identity(a.numRows, a.numRows);
        DMatrixRMaj temp = new DMatrixRMaj(a.numRows, a.numRows);
        DMatrixRMaj temp2 = new DMatrixRMaj(a.numRows, a.numRows);
        for (int i = 0; i <= exponent; ++i) {
            multOuter(a, temp);
            mult(product1, temp, temp2);
            product1.set(temp2);
        }

        // Calculate A*Omega
        DMatrixRMaj product2 = new DMatrixRMaj(a.numRows, numCols);
        mult(a, omega, product2);

        // Calculate B
        DMatrixRMaj b = new DMatrixRMaj(a.numRows, numCols);
        mult(product1, product2, b);

        QRDecomposition<DMatrixRMaj> decomposition = DecompositionFactory_DDRM.qr(b.numRows, b.numCols);
        if (!decomposition.decompose(b))
            throw new IllegalStateException("Cannot perform QR decomposition");

        DMatrixRMaj q = decomposition.getQ(null, true);
        DMatrixRMaj r = decomposition.getR(null, true);

        DMatrixRMaj s = new DMatrixRMaj(Math.min(r.numRows, r.numCols));
        extractDiag(r, s);

        DMatrixRMaj absS = new DMatrixRMaj(s.numRows, s.numCols);
        abs(s, absS);

        elementDiv(s, absS);

        DMatrixRMaj positiveQ = new DMatrixRMaj(q.numRows, q.numCols);
        mult(q, diag(s.data), positiveQ);

        return positiveQ;
    }
}
