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

import org.dulab.javanmf.measures.EuclideanDistance;
import org.dulab.javanmf.measures.Measure;
import org.ejml.data.DMatrixRMaj;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.logging.Logger;

import static org.ejml.dense.row.CommonOps_DDRM.*;

/**
 * This class performs non-negative matrix regression: for given matrices X and W, find matrix H that minimizes the
 * objective function
 * <p>
 * &emsp; D(X, WH) + &lambda;||H||<sub>1</sub> + 0.5 &mu;||H||<sup>2</sup>
 * <p>
 * where D(X, WH) is either the euclidean distance or the Kullback-Leibler divergence, ||&middot;|| is the
 * Frobenius norm, and ||&middot;||<sub>1</sub> is the <i>l</i><sub>1</sub>-norm.
 * <p>
 * <strong>Example</strong> for given matrices {@code matrixX} and {@code matrixW}, matrix {@code matrixH} is modified
 * to minimize the euclidean distance.
 * <pre> {@code
 *     UpdateRule updateRule = new MUpdateRule(0.0, 0.0);
 *
 *     MatrixRegression regression = new MatrixRegression(updateRule, 1e-6, 10000);
 *
 *     matrixH = regression.solve(matrixX, matrixW);
 * } </pre>
 *
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public class AlternatingLeastSquaresMatrixFactorization {
    /* Logger */
    private static final Logger LOG = Logger.getLogger(AlternatingLeastSquaresMatrixFactorization.class.getName());

    /* Tolerance of the fitting error */
    private final double tolerance;

    /* Maximum number of iterations */
    private final int maxIteration;

    /* Update rule */
    private final Constraint wConstraint;
    private final Constraint hConstraint;

    /* Distance measure associated with the update rule */
    private final Measure measure;

    /* Solver of the non-negative least squares problem */
    private final BroJongNonNegativeLeastSquares nonNegativeLeastSquaresForW;
    private final BroJongNonNegativeLeastSquares nonNegativeLeastSquaresForH;

    /**
     * Creates an instance of {@link AlternatingLeastSquaresMatrixFactorization}
     *
     * @param wConstraint   instance of {@link Constraint} for matrix W
     * @param hConstraint   instance of {@link Constraint} for matrix H
     * @param tolerance    the fitting error tolerance
     * @param maxIteration maximum number of iterations to use
     */
    public AlternatingLeastSquaresMatrixFactorization(@Nullable Constraint wConstraint, @Nullable Constraint hConstraint,
                                                      double tolerance, int maxIteration) {
        this.wConstraint = wConstraint != null ? wConstraint : new DefaultConstraint();
        this.hConstraint = hConstraint != null ? hConstraint : new DefaultConstraint();
        this.tolerance = tolerance;
        this.maxIteration = maxIteration;
        this.measure = new EuclideanDistance();
        this.nonNegativeLeastSquaresForW = new BroJongNonNegativeLeastSquares();
        this.nonNegativeLeastSquaresForH = new BroJongNonNegativeLeastSquares();
    }

    /**
     * Performs non-negative matrix regression with the upper limit constraint
     *
     * @param x       matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>], a collection of vectors in
     *                N<sub>points</sub>-dimensional space
     * @param w       matrix of shape [N<sub>points</sub>, N<sub>components</sub>], a collection of components
     * @param h       matrix of shape [N<sub>components</sub>, N<sub>vectors</sub>], a collection of the decomposition
     *                coefficients
     * @param verbose flag to output verbose information
     */
    public void solve(@Nonnull DMatrixRMaj x, @Nonnull DMatrixRMaj w, @Nonnull DMatrixRMaj h, boolean verbose) {

        final double initError = Math.sqrt(2 * measure.get(x, w, h));
        double prevError = initError;

        DMatrixRMaj xt = transpose(x, null);

        DMatrixRMaj wtBuffer = transpose(w, null);
        DMatrixRMaj htBuffer = new DMatrixRMaj(h.numCols, h.numRows);

        int k;
        for (k = 1; k < maxIteration + 1; ++k) {

            nonNegativeLeastSquaresForW.solve(xt, transpose(h, htBuffer), wtBuffer);
            transpose(wtBuffer, w);
            wConstraint.apply(w);
            transpose(w, wtBuffer);

            nonNegativeLeastSquaresForH.solve(x, w, h);
            hConstraint.apply(h);

            double error = Math.sqrt(2 * measure.get(x, w, h));
            double v = (prevError - error) / initError;
            if (v < tolerance) {
                if (verbose) LOG.info("NMF is completed after " + k + " iterations");
                break;
            }
            prevError = error;
        }

        if (verbose && k >= maxIteration)
            LOG.info("NMF does not converge after " + k + " iterations");
    }

    /**
     * Performs non-negative matrix regression with the upper limit constraint
     *
     * @param x matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>], a collection of vectors in
     *          N<sub>points</sub>-dimensional space
     * @param w matrix of shape [N<sub>points</sub>, N<sub>components</sub>], a collection of components
     * @param h matrix of shape [N<sub>components</sub>, N<sub>vectors</sub>], a collection of the decomposition
     *          coefficients
     */
    public void solve(@Nonnull DMatrixRMaj x, @Nonnull DMatrixRMaj w, @Nonnull DMatrixRMaj h) {
        solve(x, w, h, false);
    }
}
