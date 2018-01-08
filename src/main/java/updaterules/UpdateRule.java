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

package updaterules;

import measures.Measure;
import org.jblas.DoubleMatrix;

import javax.annotation.Nonnull;

/**
 * This class is used to update matrix H in the direction of minimization of distance between X and W x H
 *
 * @author Du-Lab Team <dulab.binf@gmail.com>
 */
public abstract class UpdateRule
{
    /* Epsilon value used for avoiding division by zero*/
    static final double EPS = 1e-12;

    /* Measure associated with the update rule */
    public final Measure measure;

    /**
     * Creates an instance of UpdateRule
     * @param measure distance measure associated with the update rule
     */
    public UpdateRule(@Nonnull Measure measure) {
        this.measure = measure;
    }

    /**
     * Updates matrix H to minimize distance between X and W x H
     * @param x matrix of shape [num_features, num_samples]
     * @param w matrix of shape [num_features, num_components]
     * @param h matrix of shape [num_components, num_samples
     * @return increment of |H|
     */
    abstract public double update(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h);
}
