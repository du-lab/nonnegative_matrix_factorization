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

package org.dulab.javanmf.updaterules;

import org.dulab.javanmf.measures.EuclideanDistance;
import org.jblas.DoubleMatrix;

import javax.annotation.Nonnull;

public class CDUpdateRule extends RegularizationUpdateRule {

    public CDUpdateRule(double lambda, double mu) {
        super(new EuclideanDistance(), lambda, mu);
    }

    @Override
    public double update(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h) {

        DoubleMatrix wt = w.transpose();

        DoubleMatrix denominator = wt.mmul(w);
        DoubleMatrix nominator = denominator.mmul(h).sub(wt.mmul(x));

        for (int r = 0; r < h.rows; ++r) {
            for (int i = 0; i < h.columns; ++i) {

                double newValue = h.get(r, i) - (nominator.get(r, i) + lambda) / denominator.get(r, r);

                h.put(r, i, newValue > 0.0 ? newValue : 0.0);
            }
        }

        return 0.0;
    }
}
