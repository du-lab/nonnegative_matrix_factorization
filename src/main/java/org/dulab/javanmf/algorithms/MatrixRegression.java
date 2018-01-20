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
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import org.dulab.javanmf.updaterules.UpdateRule;

import javax.annotation.Nonnull;
import java.util.logging.Logger;

/**
 * This class performs non-negative matrix regression: for given matrices X and W, find matrix H that minimizes the
 * objective function
 * <p>
 * &emsp; D(X, WH) + &lambda;||H||<sub>1</sub> + 0.5 &mu;||H||<sup>2</sup>
 * <p>
 *     where D(X, WH) is either the euclidean distance or the Kullback-Leibler divergence, ||&middot;|| is the
 *     Frobenius norm, and ||&middot;||<sub>1</sub> is the <i>l</i><sub>1</sub>-norm.
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
public class MatrixRegression
{
    /* Logger */
    private static final Logger LOG = Logger.getLogger(MatrixRegression.class.getName());

    /* Tolerance of the fitting error */
    private final double tolerance;

    /* Maximum number of iterations */
    private final int maxIteration;

    /* Update rule */
    private final UpdateRule updateRule;

    /* Distance measure associated with the update rule */
    private final Measure measure;

    /**
     * Creates an instance of {@link MatrixRegression}
     * @param updateRule instance of {@link UpdateRule} for matrix H
     * @param tolerance the fitting error tolerance
     * @param maxIteration maximum number of iterations to use
     */
    public MatrixRegression(@Nonnull UpdateRule updateRule, double tolerance, int maxIteration) {
        this.updateRule = updateRule;
        this.measure = updateRule.measure;
        this.tolerance = tolerance;
        this.maxIteration = maxIteration;
    }

    /**
     * Performs non-negative matrix regression with the upper limit constraint
     * @param x matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>], a collection of vectors in
     *          N<sub>points</sub>-dimensional space
     * @param w matrix of shape [N<sub>points</sub>, N<sub>components</sub>], a collection of components
     * @param limit matrix of shape [N<sub>components</sub>, N<sub>vectors</sub>], the upper limit for matrix H
     * @param verbose flag to output verbose information
     * @return matrix H of shape [N<sub>components</sub>, N<sub>vectors</sub>]
     */
    public DoubleMatrix solve(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix limit,
                              boolean verbose)
    {
        DoubleMatrix h = Solve.solveLeastSquares(w, x).max(0.0).min(limit);

        final double initError = Math.sqrt(2 * measure.get(x, w, h));
        double prevError = initError;

        int k;
        for (k = 1; k < maxIteration + 1; ++k)
        {
            updateRule.update(x, w, h);
            h.mini(limit);

            if (k % 10 == 0) {
                double error = Math.sqrt(2 * measure.get(x, w, h));
                if ((prevError - error) / initError < tolerance) {
                    if (verbose) LOG.info("NLS is completed after " + k + " iterations");
                    break;
                }
                prevError = error;
            }
        }

        if (verbose && k >= maxIteration)
            LOG.info("NLS does not converge after " + k + " iterations");

        return h;
    }

    /**
     * Performs non-negative matrix regression with the upper limit constraint
     * @param x matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>], a collection of vectors in
     *          N<sub>points</sub>-dimensional space
     * @param w matrix of shape [N<sub>points</sub>, N<sub>components</sub>], a collection of components
     * @param limit matrix of shape [N<sub>components</sub>, N<sub>vectors</sub>], the upper limit for matrix H
     * @return matrix H of shape [N<sub>components</sub>, N<sub>vectors</sub>]
     */
    public DoubleMatrix solve(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix limit) {
        return solve(x, w, limit, false);
    }

    /**
     * Performs non-negative matrix regression
     * @param x matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>], a collection of vectors in N<sub>points</sub>-dimensional space
     * @param w matrix of shape [N<sub>points</sub>, N<sub>components</sub>], a collection of components
     * @param verbose flag to output verbose information
     * @return matrix H of shape [N<sub>components</sub>, N<sub>vectors</sub>]
     */
    public DoubleMatrix solve(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, boolean verbose)
    {
        DoubleMatrix limit = DoubleMatrix.ones(w.columns, x.columns).mul(Double.MAX_VALUE);
        return solve(x, w, limit, verbose);
    }

    /**
     * Performs non-negative matrix regression
     * @param x matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>], a collection of vectors in N<sub>points</sub>-dimensional space
     * @param w matrix of shape [N<sub>points</sub>, N<sub>components</sub>], a collection of components
     * @return matrix H of shape [N<sub>components</sub>, N<sub>vectors</sub>]
     */
    public DoubleMatrix solve(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w) {
        return solve(x, w, false);
    }
}
