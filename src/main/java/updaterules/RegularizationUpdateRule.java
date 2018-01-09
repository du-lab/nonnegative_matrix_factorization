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

package updaterules;

import measures.Measure;

import javax.annotation.Nonnull;

/**
 * Provides a template for updating matrix H in the direction of minimizing the distance D(X, WH) with regularization
 *
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public abstract class RegularizationUpdateRule extends UpdateRule
{
    /** <i>l</i><sub>1</sub>-regularization coefficient */
    protected final double lambda;

    /** <i>l</i><sub>2</sub>-regularization coefficient */
    protected final double mu;

    /**
     * Creates an instance of {@link RegularizationUpdateRule} with given regularization coefficients
     * @param measure distance measure associated with the update rule
     * @param lambda <i>l</i><sub>1</sub>-regularization coefficient
     * @param mu <i>l</i><sub>2</sub>-regularization coefficient
     * @throws IllegalArgumentException If at least one of the regularization coefficients is negative
     */
    public RegularizationUpdateRule(@Nonnull Measure measure, double lambda, double mu) throws IllegalArgumentException
    {
        super(measure);

        if (lambda < 0.0)
            throw new IllegalArgumentException("Negative l1-regularization coefficient " + lambda);

        if (mu < 0.0)
            throw new IllegalArgumentException("Negative l2-regularization coefficient " + mu);

        this.lambda = lambda;
        this.mu = mu;
    }
}
