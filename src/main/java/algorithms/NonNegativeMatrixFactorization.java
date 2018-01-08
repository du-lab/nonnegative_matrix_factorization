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

package algorithms;

import measures.Measure;
import org.jblas.*;
import updaterules.UpdateRule;

import javax.annotation.Nonnull;
import java.util.logging.Logger;

/**
 * This class performs non-negative matrix factorization.
 * <p>
 * Given a matrix X with non-negative entries, find non-negative matrices W and H that minimize the objective
 * function <em>Distance(X, W x H)</em>
 *
 * @author Du-Lab Team <dulab.binf@gmail.com>
 */
public class NonNegativeMatrixFactorization
{
    /* Logger */
    private static final Logger LOG = Logger.getLogger(NonNegativeMatrixFactorization.class.getName());

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
     * Creates an instance of Non-Negative Matrix Factorization
     * @param updateRuleW instance of {@link updaterules.UpdateRule} for matrix W
     * @param updateRuleH instance of {@link updaterules.UpdateRule} for matrix H
     * @param tolerance tolerance for the fitting error
     * @param maxIteration maximum number of iterations to use
     */
    public NonNegativeMatrixFactorization(@Nonnull UpdateRule updateRuleW, @Nonnull UpdateRule updateRuleH,
            double tolerance, int maxIteration)
    {
        this.updateRuleW = updateRuleW;
        this.updateRuleH = updateRuleH;
        this.measure = updateRuleW.measure;
        this.tolerance = tolerance;
        this.maxIteration = maxIteration;
    }

    /**
     * Performs the non-negative matrix factorization with given initial components and coefficients.
     * We use alternate coordinate descent method where we alternate between updating matrices W and H
     *
     * @param data matrix of shape [num_points, num_vectors], a collection of vectors in num_points-dimensional space
     * @param w matrix of shape [num_points, num_vectors], a collection of initial components
     * @param h matrix of shape [num_components, num_vectors], a collection of initial decomposition coefficients
     */
    public void execute(@Nonnull DoubleMatrix data, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h)
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
                    LOG.info("NMF is completed after " + k + " iterations");
                    break;
                }
                prevError = error;
            }
        }

        if (k >= maxIteration)
            LOG.info("NMF does not converge after " + k + " iterations");

        w.copy(wt.transpose());
    }
}
