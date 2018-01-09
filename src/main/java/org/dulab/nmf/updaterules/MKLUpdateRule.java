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

import org.dulab.nmf.measures.KLDivergence;
import org.jblas.DoubleMatrix;

import javax.annotation.Nonnull;

/**
 * Performs multiplicative update for the generalized Kullback-Leibler divergence with regularization
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public class MKLUpdateRule extends RegularizationUpdateRule
{
    /**
     * Creates an instance of {@link MKLUpdateRule} with given regularization coefficients
     * @param lambda <i>l</i><sub>1</sub>-regularization coefficient
     * @param mu <i>l</i><sub>2</sub>-regularization coefficient
     */
    public MKLUpdateRule(double lambda, double mu) {
        super(new KLDivergence(), lambda, mu);
    }

    @Override
    public double update(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h)
    {
        DoubleMatrix wt = w.transpose();
        h.muli(wt.mmul(x.div(w.mmul(h).max(EPS))).div(
                h.mul(mu).add(lambda).addColumnVector(wt.rowSums()).max(EPS)));
        return 0.0;
    }
}
