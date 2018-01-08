/*
 * Copyright (C) 2017 Du-Lab Team <dulab.binf@gmail.com>
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

import measures.Measure;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import updaterules.UpdateRule;

import javax.annotation.Nonnull;
import java.util.logging.Logger;

/**
 * This class performs Non-Negative Least-Squares optimization.
 *
 * For matrices X and W with non-negative entries, we find non-negative matrix H such that
 *
 *     X = dot(W, H)
 *
 *   > X of shape [num_points, num_vectors] is a collection of vectors in num_points-dimensional space
 *   > W of shape [num_points, num_components] is a collection of independent components that best describe the vectors
 *   > H of shape [num_components, num_vectors] is a mixing matrix that maps components to vectors
 *
 * @author Du-Lab Team <dulab.binf@gmail.com>
 */
public class NonNegativeLeastSquares
{
    /* Logger */
    private static final Logger LOG = Logger.getLogger(NonNegativeLeastSquares.class.getName());

    /* Tolerance of the fitting error */
    private final double tolerance;

    /* Maximum number of iterations */
    private final int maxIteration;

    /* Update rule */
    private final UpdateRule updateRule;

    /* Distance measure associated with the update rule */
    private final Measure measure;

    public NonNegativeLeastSquares(@Nonnull UpdateRule updateRule, double tolerance, int maxIteration) {
        this.updateRule = updateRule;
        this.measure = updateRule.measure;
        this.tolerance = tolerance;
        this.maxIteration = maxIteration;
    }

    public DoubleMatrix solve(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix limit)
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
                    LOG.info("NLS is completed after " + k + " iterations");
                    break;
                }
                prevError = error;
            }
        }

        if (k >= maxIteration)
            LOG.info("NLS does not converge after " + k + " iterations");

        return h;
    }

    @Nonnull
    public DoubleMatrix solve(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w)
    {
        DoubleMatrix limit = DoubleMatrix.ones(w.columns, x.columns).mul(Double.MAX_VALUE);
        return solve(x, w, limit);
    }
}
