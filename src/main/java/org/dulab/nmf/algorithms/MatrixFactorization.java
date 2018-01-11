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

package org.dulab.nmf.algorithms;

import org.dulab.nmf.measures.Measure;
import org.jblas.*;
import org.dulab.nmf.updaterules.UpdateRule;

import javax.annotation.Nonnull;
import java.util.logging.Logger;

/**
 * This class performs non-negative matrix factorization: for given matrix X, find matrices W and H that minimize the
 * objective function
 * <p>
 * &emsp; D(X, WH) + &lambda;<sub>w</sub>||W||<sub>1</sub> + &lambda;<sub>h</sub>||H||<sub>1</sub> +
 * 0.5 &mu;<sub>w</sub>||W||<sup>2</sup> + 0.5 &mu;<sub>h</sub>||H||<sup>2</sup>
 * <p>
 *     where D(X, WH) is either the euclidean distance or the Kullback-Leibler divergence, ||&middot;|| is the
 *     Frobenius norm, and ||&middot;||<sub>1</sub> is the <i>l</i><sub>1</sub>-norm.
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
public class MatrixFactorization
{
    /* Logger */
    private static final Logger LOG = Logger.getLogger(MatrixFactorization.class.getName());

    /** Tolerance of the fitting error */
    private final double tolerance;

    /** Maximum number of iterations */
    private final int maxIteration;

    /* Update rule for matrix W */
    private final UpdateRule updateRuleW;

    /* Update rule for matrix H */
    private final UpdateRule updateRuleH;

    /* Distance measure associated with the update rules */
    private final Measure measure;

    /**
     * Creates an instance of {@link MatrixFactorization}
     * @param updateRuleW instance of {@link org.dulab.nmf.updaterules.UpdateRule} for matrix W
     * @param updateRuleH instance of {@link org.dulab.nmf.updaterules.UpdateRule} for matrix H
     * @param tolerance the fitting error tolerance
     * @param maxIteration maximum number of iterations to use
     */
    public MatrixFactorization(@Nonnull UpdateRule updateRuleW, @Nonnull UpdateRule updateRuleH,
                               double tolerance, int maxIteration)
    {
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
     * @param data matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>], a collection of vectors in N<sub>points</sub>-dimensional space
     * @param w matrix of shape [N<sub>points</sub>, N<sub>components</sub>], a collection of initial components
     * @param h matrix of shape [N<sub>components</sub>, N<sub>vectors</sub>], a collection of initial coefficients
     * @param verbose flag to output verbose information
     */
    public void execute(@Nonnull DoubleMatrix data, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h, boolean verbose)
    {
        DoubleMatrix x = data.dup();
        DoubleMatrix xt = x.transpose();
        DoubleMatrix wt = w.transpose();

        final double initError = measure.get(x, w, h);
        double prevError = initError;

        // Update matrices WT and H until the error is small or the maximum number of iterations is reached
        int k;
        for (k = 1; k < maxIteration + 1; ++k)
        {
            updateRuleW.update(xt, h.transpose(), wt);
            updateRuleH.update(x, wt.transpose(), h);

            if (k % 10 == 0) {
                double error = measure.get(x, wt.transpose(), h);
                if ((prevError - error) / initError < tolerance) {
                    if (verbose) LOG.info("NMF is completed after " + k + " iterations");
                    break;
                }
                prevError = error;
            }
        }

        if (verbose && k >= maxIteration)
            LOG.info("NMF does not converge after " + k + " iterations");

        w.copy(wt.transpose());
    }

    /**
     * Performs the non-negative matrix factorization with given initial matrices W and H.
     * <p>
     * Parameters {@code w} and {@code h} contain the result of the factorization.
     *
     * @param data matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>], a collection of vectors in
     *             N<sub>points</sub>-dimensional space
     * @param w matrix of shape [N<sub>points</sub>, N<sub>components</sub>], a collection of initial components
     * @param h matrix of shape [N<sub>components</sub>, N<sub>vectors</sub>], a collection of initial coefficients
     */
    public void execute(@Nonnull DoubleMatrix data, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h) {
        execute(data, w, h, false);
    }
}
