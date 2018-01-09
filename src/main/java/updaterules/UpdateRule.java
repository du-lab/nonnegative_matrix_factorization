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
 * Provides a template for updating matrix H in the direction of minimizing the distance D(X, WH)
 *
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public abstract class UpdateRule
{
    /* Epsilon value used for avoiding division by zero*/
    static final double EPS = 1e-12;

    /** Instance of {@link Measure} associated with this update rule */
    public final Measure measure;

    /**
     * Creates an instance of {@link UpdateRule}
     * @param measure instance of {@link Measure} associated with the update rule
     */
    public UpdateRule(@Nonnull Measure measure) {
        this.measure = measure;
    }

    /**
     * Updates matrix H to minimize distance between X and WH
     * @param x matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>]
     * @param w matrix of shape [N<sub>points</sub>, N<sub>components</sub>]
     * @param h matrix of shape [N<sub>components</sub>, N<sub>vectors</sub>]
     * @return increment of |H|
     */
    abstract public double update(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h);
}
