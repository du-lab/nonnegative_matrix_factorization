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
import org.ejml.data.DMatrixRMaj;
import org.dulab.javanmf.updaterules.UpdateRule;

import javax.annotation.Nonnull;
import java.util.logging.Logger;

import static org.ejml.dense.row.CommonOps_DDRM.*;

/**
 * This class performs non-negative matrix factorization: for given matrix X, find matrices W and H that minimize the
 * objective function
 * <p>
 * &emsp; D(X, WH) + &lambda;<sub>w</sub>||W||<sub>1</sub> + &lambda;<sub>h</sub>||H||<sub>1</sub> +
 * 0.5 &mu;<sub>w</sub>||W||<sup>2</sup> + 0.5 &mu;<sub>h</sub>||H||<sup>2</sup>
 * <p>
 * where D(X, WH) is either the euclidean distance or the Kullback-Leibler divergence, ||&middot;|| is the
 * Frobenius norm, and ||&middot;||<sub>1</sub> is the <i>l</i><sub>1</sub>-norm.
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
public class MatrixFactorization {
    /* Logger */
    private static final Logger LOG = Logger.getLogger(MatrixFactorization.class.getName());

    /**
     * Tolerance of the fitting error
     */
    private final double tolerance;

    /**
     * Maximum number of iterations
     */
    private final int maxIteration;

    /* Update rule for matrix W */
    private final UpdateRule updateRuleW;

    /* Update rule for matrix H */
    private final UpdateRule updateRuleH;

    /* Distance measure associated with the update rules */
    private final Measure measure;

    /**
     * Creates an instance of {@link MatrixFactorization}
     *
     * @param updateRuleW  instance of {@link org.dulab.javanmf.updaterules.UpdateRule} for matrix W
     * @param updateRuleH  instance of {@link org.dulab.javanmf.updaterules.UpdateRule} for matrix H
     * @param tolerance    the fitting error tolerance
     * @param maxIteration maximum number of iterations to use
     */
    public MatrixFactorization(@Nonnull UpdateRule updateRuleW, @Nonnull UpdateRule updateRuleH,
                               double tolerance, int maxIteration) {
        this.updateRuleW = updateRuleW;
        this.updateRuleH = updateRuleH;
        this.measure = updateRuleW.measure;
        this.tolerance = tolerance;
        this.maxIteration = maxIteration;
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

        DMatrixRMaj xt = new DMatrixRMaj(x.numCols, x.numRows);
        transpose(x, xt);

        DMatrixRMaj wt = new DMatrixRMaj(w.numCols, w.numRows);
        transpose(w, wt);

//        DoubleMatrix xt = x.transpose();
//        DoubleMatrix wt = w.transpose();

        final double initError = measure.get(x, w, h);
        double prevError = initError;

        DMatrixRMaj htBuffer = new DMatrixRMaj(h.numCols, h.numRows);
        DMatrixRMaj wttBuffer = new DMatrixRMaj(w.numRows, w.numCols);
//        DoubleMatrix htBuffer = new DoubleMatrix();
//        DoubleMatrix wttBuffer = new DoubleMatrix();

        // Update matrices WT and H until the error is small or the maximum number of iterations is reached
        int k;
        for (k = 1; k < maxIteration + 1; ++k) {

            updateRuleH.update(x, transpose(wt, wttBuffer), h);
            updateRuleW.update(xt, transpose(h, htBuffer), wt);

            if (k % 10 == 0) {
                double error = measure.get(x, transpose(wt, wttBuffer), h);
                if (Math.abs(prevError - error) / initError < tolerance) {
                    if (verbose) LOG.fine("NMF is completed after " + k + " iterations");
                    break;
                }
                prevError = error;
            }
        }

        if (verbose && k >= maxIteration)
            LOG.fine("NMF does not converge after " + k + " iterations");

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
}
