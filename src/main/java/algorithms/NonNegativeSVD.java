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

package algorithms;

import org.jblas.DoubleMatrix;
import org.jblas.Singular;

import javax.annotation.Nonnull;

/**
 * @author Du-Lab Team <dulab.binf@gmail.com>
 */
public class NonNegativeSVD
{
    private final DoubleMatrix matrixU;
    private final DoubleMatrix vectorS;
    private final DoubleMatrix matrixV;

    private final int wLength;
    private final int hLength;

    public NonNegativeSVD(@Nonnull DoubleMatrix matrix)
    {
        DoubleMatrix[] svd = Singular.fullSVD(matrix);
        matrixU = svd[0];
        vectorS = svd[1];
        matrixV = svd[2];

        wLength = matrixU.columns;
        hLength = matrixV.columns;
    }

    public Pair getPair(int index)
    {
        if (index < 0 || index >= vectorS.length)
            throw new IllegalArgumentException("Index " + index + " is out of range");

        double uPositiveNorm = columnPositiveNorm2(matrixU, index);
        double vPositiveNorm = columnPositiveNorm2(matrixV, index);
        double mp = uPositiveNorm * vPositiveNorm;

        double uNegativeNorm = columnNegativeNorm2(matrixU, index);
        double vNegativeNorm = columnNegativeNorm2(matrixV, index);
        double mn = uNegativeNorm * vNegativeNorm;

        Pair pair = new Pair(wLength, hLength);

        if (mp > mn) {
            double sqrtS = Math.sqrt(vectorS.get(index) * mp);

            for (int i = 0; i < pair.w.length; ++i) {
                double value = Math.max(matrixU.get(i, index), 0.0);
                value /= uPositiveNorm;
                pair.w[i] = sqrtS * value;
            }

            for (int i = 0; i < pair.h.length; ++i) {
                double value = Math.max(matrixV.get(i, index), 0.0);
                value /= vPositiveNorm;
                pair.h[i] = sqrtS * value;
            }
        }
        else {
            double sqrtS = Math.sqrt(vectorS.get(index) * mn);

            for (int i = 0; i < pair.w.length; ++i) {
                double value = -Math.min(matrixU.get(i, index), 0.0);
                value /= uNegativeNorm;
                pair.w[i] = sqrtS * value;
            }

            for (int i = 0; i < pair.h.length; ++i) {
                double value = -Math.min(matrixV.get(i, index), 0.0);
                value /= vNegativeNorm;
                pair.h[i] = sqrtS * value;
            }
        }

        return pair;
    }

    public void decompose(@Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h)
            throws IllegalArgumentException
    {
        if (w.columns != h.rows)
            throw new IllegalArgumentException("Cannot perform SVD decomposition");

        for (int j = 0; j < w.columns; ++j)
            calculate(w, h, j);
    }

    public void calculate(@Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h, int index)
    {
        if (index < 0 || index >= vectorS.length)
            throw new IllegalArgumentException("Index " + index + " is out of range");

        double uPositiveNorm = columnPositiveNorm2(matrixU, index);
        double vPositiveNorm = columnPositiveNorm2(matrixV, index);
        double mp = uPositiveNorm * vPositiveNorm;

        double uNegativeNorm = columnNegativeNorm2(matrixU, index);
        double vNegativeNorm = columnNegativeNorm2(matrixV, index);
        double mn = uNegativeNorm * vNegativeNorm;

        if (mp > mn) {
            double sqrtS = Math.sqrt(vectorS.get(index) * mp);

            for (int i = 0; i < w.rows; ++i) {
                double value = Math.max(matrixU.get(i, index), 0.0);
                value /= uPositiveNorm;
                w.put(i, index, sqrtS * value);
            }

            for (int i = 0; i < h.columns; ++i) {
                double value = Math.max(matrixV.get(i, index), 0.0);
                value /= vPositiveNorm;
                h.put(index, i, sqrtS * value);
            }
        }
        else {
            double sqrtS = Math.sqrt(vectorS.get(index) * mn);

            for (int i = 0; i < w.rows; ++i) {
                double value = -Math.min(matrixU.get(i, index), 0.0);
                value /= uNegativeNorm;
                w.put(i, index, sqrtS * value);
            }

            for (int i = 0; i < h.columns; ++i) {
                double value = -Math.min(matrixV.get(i, index), 0.0);
                value /= vNegativeNorm;
                h.put(index, i, sqrtS * value);
            }
        }
    }

    public static class Pair {
        public final double[] w;
        public final double[] h;

        public Pair(int wLength, int hLength) {
            w = new double[wLength];
            h = new double[hLength];
        }
    }

    /**
     * Calculates l2-norm of positive elements of a column
     * @param m matrix
     * @param col index of a column
     * @return l2-norm of positive elements of the vector
     */
    private double columnPositiveNorm2(@Nonnull DoubleMatrix m, int col) {
        double sum = 0.0;
        for (int i = 0; i < m.rows; ++i) {
            double v = m.get(i, col);
            if (v > 0.0)
                sum += v * v;
        }
        return java.lang.Math.sqrt(sum);
    }

    /**
     * Calculates l2-norm of negative elements of a column
     * @param m matrix
     * @param col index of a column
     * @return l2-norm of negative elements of the vector
     */
    private double columnNegativeNorm2(@Nonnull DoubleMatrix m, int col) {
        double sum = 0.0;
        for (int i = 0; i < m.rows; ++i) {
            double v = m.get(i, col);
            if (v < 0.0)
                sum += v * v;
        }
        return Math.sqrt(sum);
    }
}
