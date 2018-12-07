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

package org.dulab.javanmf.updaterules;

import org.dulab.javanmf.measures.EuclideanDistance;
import org.jblas.DoubleMatrix;

import javax.annotation.Nonnull;
import java.util.stream.IntStream;

/**
 * Performs multiplicative update for the euclidean distance with regularization
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public class MUpdateRule extends RegularizationUpdateRule
{
    /**
     * Creates an instance of {@link MUpdateRule} with given regularization coefficients
     * @param lambda <i>l</i><sub>1</sub>-regularization coefficient
     * @param mu <i>l</i><sub>2</sub>-regularization coefficient
     */
    public MUpdateRule(double lambda, double mu) {
        super(new EuclideanDistance(), lambda, mu);
    }

    @Override
    public double update(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h)
    {
        double a = x.length;
        double b = h.length;

        DoubleMatrix wt = w.transpose();
//        h.muli(wt.mmul(x).div(wt.mmul(w).mmul(h).add(a / b * lambda).add(h.mul(a / b * mu)).max(1e-12)));

        h.muli(wt.mmul(x).divi(wt.mmul(w).mmul(h).addi(a / b * lambda).addi(h.mul(a / b * mu)).maxi(1e-12)));

//        DoubleMatrix buffer1 = new DoubleMatrix(h.rows, h.columns);
//        h.muli(wt.mmuli(x, buffer1));
//
//        DoubleMatrix buffer2 = new DoubleMatrix(w.columns, w.columns);
//        wt.mmuli(w, buffer2);
//        buffer2.mmuli(h, buffer1);
//        buffer1.addi(a / b * lambda);
//        buffer1.addi(h.mul(a / b * mu));
//        buffer1.maxi(1e-12);
//        h.divi(buffer1);

//        for (int i = 0; i < h.rows; ++i) {
//            for (int j = 0; j < h.columns; ++j) {
//
//                double nominator = 0.0;
//                for (int k = 0; k < x.rows; ++k) {
//                    nominator += w.get(k, i) * x.get(k, j);
//                }
//
//                double denominator = 0.0;
//                for (int k = 0; k < w.rows; ++k) {
//                    for (int l = 0; l < h.rows; ++l) {
//                        denominator += w.get(k, i) * w.get(k, l) * h.get(l, j);
//                    }
//                }
//                denominator += a / b * lambda;
//                denominator += a / b * mu * h.get(i, j);
//                denominator = denominator > 1e-12 ? denominator : 1e-12;
//
//                h.put(i, j, h.get(i, j) * nominator / denominator);
//            }
//        }

        return 0.0;
    }
}
