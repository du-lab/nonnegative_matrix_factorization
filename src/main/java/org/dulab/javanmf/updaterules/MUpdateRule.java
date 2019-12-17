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
import org.ejml.data.DMatrixRMaj;

import javax.annotation.Nonnull;

import static org.ejml.dense.row.CommonOps_DDRM.*;

/**
 * Performs multiplicative update for the euclidean distance with regularization
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public class MUpdateRule extends RegularizationUpdateRule
{
    private DMatrixRMaj wtx = null;
    private DMatrixRMaj wtw = null;
    private DMatrixRMaj wtwh = null;

    /**
     * Creates an instance of {@link MUpdateRule} with given regularization coefficients
     * @param lambda <i>l</i><sub>1</sub>-regularization coefficient
     * @param mu <i>l</i><sub>2</sub>-regularization coefficient
     */
    public MUpdateRule(double lambda, double mu) {
        super(new EuclideanDistance(), lambda, mu);
    }

    @Override
    public double update(@Nonnull DMatrixRMaj x, @Nonnull DMatrixRMaj w, @Nonnull DMatrixRMaj h)
    {
        double a = x.getNumElements();
        double b = h.getNumElements();

//        DoubleMatrix wt = w.transpose();
//        h.muli(wt.mmul(x).div(wt.mmul(w).mmul(h).add(a / b * lambda).add(h.mul(a / b * mu)).max(1e-12)));

        if (wtx == null || wtx.numRows != w.numCols || wtx.numCols != x.numCols)
            wtx = new DMatrixRMaj(w.numCols, x.numCols);

        if (wtw == null || wtw.numRows != w.numCols || wtw.numCols != w.numCols)
            wtw = new DMatrixRMaj(w.numCols, w.numCols);

        if (wtwh == null || wtwh.numRows != w.numCols || wtwh.numCols != h.numCols)
            wtwh = new DMatrixRMaj(w.numCols, h.numCols);

        // Nominator
        multTransA(w, x, wtx);  // wtx = wt * x

        // Denominator
        multTransA(w, w, wtw);  // wtw = wt * w
        mult(wtw, h, wtwh);  // wtwh = wt * w * h
        add(wtwh, lambda * a / b);  // wtwh = wt * w * h + lambda * a / b
        addEquals(wtwh, mu * a / b, h);  // wtwh = wt * w * h + lambda * a / b + mu * a / b * h
        add(wtwh, 1e-12);  // wtwh = wt * w * h + lambda * a / b + mu * a / b * h + 1e-12

        // Fraction
        elementDiv(wtx, wtwh);  // wtx = wtx (/) wtwh
        elementMult(h, wtx);  // h = h (*) wtx

        return 0.0;
    }
}
