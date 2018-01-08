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
 * This class is used to update matrix H in the direction of minimization of regularized distance between X and W x H
 * @author Du-Lab Team <dulab.binf@gmail.com>
 */
public abstract class RegularizationUpdateRule extends UpdateRule
{
    /* l1-regularization coefficient */
    protected final double lambda;

    /* l2-regularization coefficient */
    protected final double mu;

    /**
     * Creates an instance of RegularizationUpdateRule with given regularization coefficients
     * @param measure distance measure associated with the update rule
     * @param lambda l1-regularization coefficient
     * @param mu l2-regularization coefficient
     * @throws IllegalArgumentException When at least one of the regularization coefficients is negative
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
