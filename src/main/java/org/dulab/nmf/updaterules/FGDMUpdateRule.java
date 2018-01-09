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

package org.dulab.nmf.updaterules;

import org.dulab.nmf.measures.EuclideanDistance;
import org.jblas.DoubleMatrix;

import javax.annotation.Nonnull;

/**
 * Performs fast-gradient-descent update for the euclidean distance with regularization
 *
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public class FGDMUpdateRule extends RegularizationUpdateRule
{
    /**
     * Creates an instance of FGDMUpdateRule with given regularization coefficients
     * @param lambda <i>l</i><sub>1</sub>-regularization coefficient
     * @param mu <i>l</i><sub>2</sub>-regularization coefficient
     */
    public FGDMUpdateRule(double lambda, double mu) throws IllegalArgumentException {
        super(new EuclideanDistance(), lambda, mu);
    }

    @Override
    public double update(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h)
    {
        final int numFeatures = x.rows;
        final int numSamples = x.columns;
        final int numComponents = w.columns;

        // Calculate the negative scaled gradient Nabla
        DoubleMatrix nabla = getNabla(x, w, h);

        // Find the optimal rate of convergence Theta
        DoubleMatrix wd = w.mmul(nabla);
        DoubleMatrix wh = w.mmul(h);
        DoubleMatrix theta = DoubleMatrix.ones(1, numSamples);
        for (int t = 0; t < 100; ++t)
        {
            double norm = 0.0;
            for (int j = 0; j < numSamples; ++j) {
                double a = 0.0;
                double dphi = 0.0;
                for (int l = 0; l < numFeatures; ++l) {
                    double fraction = wd.get(l, j) / (wh.get(l, j) + theta.get(j) * wd.get(l, j));
                    if (Double.isFinite(fraction)) {
                        a += wd.get(l, j) * wd.get(l, j);  // Hessian of phi(theta)
                        dphi -= (x.get(l, j) - wh.get(l, j) - theta.get(j) * wd.get(l, j)) * wd.get(l, j);  // Derivative of phi(theta)
                    }
                }

                for (int k = 0; k < numComponents; ++k) {
                    a += mu * nabla.get(k, j) * nabla.get(k, j);
                    dphi += lambda * nabla.get(k, j)
                            + mu * h.get(k, j) * nabla.get(k, j)
                            + mu * theta.get(j) * nabla.get(k, j) * nabla.get(k, j);
                }

                double d = dphi / a;
                if (Double.isFinite(d) && -1 < d && d < theta.get(j)) {
                    norm += d * d;
                    theta.put(j, theta.get(j) - d);
                }
            }
            norm = Math.sqrt(norm);

            if (norm < 1e-4) break;
        }

        // Correct Theta so that H would stay in the range [0, upperLimit]
        double min = 1.0;
        for (int j = 0; j < numSamples; ++j) {
            double supD = Double.MAX_VALUE;
            for (int i = 0; i < numComponents; ++i)
                if (nabla.get(i, j) < 0.0 && -h.get(i, j) / nabla.get(i, j) < supD)
                    supD = -h.get(i, j) / nabla.get(i, j);

            double d = (supD - 1.0) / (theta.get(j) - 1.0);
            if (theta.get(j) > supD && d < min)
                min = d;
        }
        double alpha = 0.99 * min;
        theta = theta.mul(alpha).add(1.0 - alpha);

        // Update matrix H
        nabla.muliRowVector(theta);
        double delta = nabla.norm2() / h.norm2();
        h.addi(nabla);

        return delta;
    }

    /**
     * Calculates the scaled negative gradient Nabla
     * @param x matrix of shape [num_features, num_samples]
     * @param w matrix of shape [num_features, num_components]
     * @param h matrix of shape [num_components, num_samples
     * @return Nabla
     */
    private DoubleMatrix getNabla(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h)
    {
        DoubleMatrix wt = w.transpose();
        DoubleMatrix nabla = wt.mmul(x).div(wt.mmul(w).mmul(h).add(lambda).add(h.mul(mu)).max(EPS));
        nabla.muli(h);
        nabla.subi(h);

        return nabla;
    }
}
