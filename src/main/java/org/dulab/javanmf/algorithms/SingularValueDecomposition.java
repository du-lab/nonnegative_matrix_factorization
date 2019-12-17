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

package org.dulab.javanmf.algorithms;

import org.ejml.data.DMatrixRMaj;
import org.ejml.interfaces.decomposition.SingularValueDecomposition_F64;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;

import javax.annotation.Nonnull;

/**
 * This class performs non-negative singular value decomposition: first, the singular value decomposition is
 * performed; then, the non-negative matrices W and H are formed.
 * <p>
 * Based on <a href="http://www.sciencedirect.com/science/article/pii/S0031320307004359">C. Boutsidis and E. Gallopoulos,
 *     SVD based initialization: A head start for nonnegative matrix factorization</a>
 * <p>
 * <strong>Examples</strong> for given matrix X and number of components N<sub>components</sub>
 *
 * <pre> {@code
 *     final int num_points = matrixX.numRows;
 *     final int num_vectors = matrixX.numCols;
 *
 *     DoubleMatrix matrixW = new DoubleMatrix(num_points, num_components);
 *     DoubleMatrix matrixH = new DoubleMatrix(num_components, num_vectors);
 *
 *     SingularValueDecomposition decomposition = new SingularValueDecomposition(matrixX);
 *     decomposition.decompose(matrixW, matrixH);
 * } </pre>
 *
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public class SingularValueDecomposition
{
    private final DMatrixRMaj matrixU;
    private final DMatrixRMaj vectorS;
    private final DMatrixRMaj matrixV;

    /**
     * Creates an instance of {@link SingularValueDecomposition} for given {@code matrix}
     * @param x matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>] to be decomposed
     */
    public SingularValueDecomposition(@Nonnull DMatrixRMaj x)
    {
        SingularValueDecomposition_F64<DMatrixRMaj> svd =
                DecompositionFactory_DDRM.svd(x.numRows, x.numCols, true, true, false);

        if (!svd.decompose(x))
            throw new IllegalStateException("Decomposition failed");

        matrixU = svd.getU(null, false);
        vectorS = svd.getW(null);
        matrixV = svd.getV(null, false);
    }

    /**
     * Performs non-negative singular value decomposition (NNDSVD) of matrix X.
     * <p>
     * The parameters {@code w} and {@code h} contain the result of the decomposition.
     * <p>
     * For details, see <a href="http://www.sciencedirect.com/science/article/pii/S0031320307004359">C. Boutsidis and
     * E. Gallopoulos, SVD based initialization: A head start for nonnegative matrix factorization</a>.
     *
     * @param w matrix of shape [N<sub>points</sub>, N<sub>components</sub>]
     * @param h matrix of shape [N<sub>components</sub>, N<sub>vectors</sub>]
     * @throws IllegalArgumentException if the number of columns in matrix X is not equal to the number of rows in
     * matrix H
     */
    public void decompose(@Nonnull DMatrixRMaj w, @Nonnull DMatrixRMaj h)
            throws IllegalArgumentException
    {
        if (w.numCols != h.numRows)
            throw new IllegalArgumentException("Cannot perform SVD decomposition");

        for (int j = 0; j < w.numCols; ++j)
            calculate(w, h, j);
    }

    private void calculate(@Nonnull DMatrixRMaj w, @Nonnull DMatrixRMaj h, int index)
    {
        if (index < 0 || index >= vectorS.getNumElements())
            throw new IllegalArgumentException("Index " + index + " is out of range");

        double uPositiveNorm = columnPositiveNorm2(matrixU, index);
        double vPositiveNorm = columnPositiveNorm2(matrixV, index);
        double mp = uPositiveNorm * vPositiveNorm;

        double uNegativeNorm = columnNegativeNorm2(matrixU, index);
        double vNegativeNorm = columnNegativeNorm2(matrixV, index);
        double mn = uNegativeNorm * vNegativeNorm;

        if (mp > mn) {
            double sqrtS = Math.sqrt(vectorS.get(index) * mp);

            for (int i = 0; i < w.numRows; ++i) {
                double value = Math.max(matrixU.get(i, index), 0.0);
                value /= uPositiveNorm;
                w.set(i, index, sqrtS * value);
            }

            for (int i = 0; i < h.numCols; ++i) {
                double value = Math.max(matrixV.get(i, index), 0.0);
                value /= vPositiveNorm;
                h.set(index, i, sqrtS * value);
            }
        }
        else {
            double sqrtS = Math.sqrt(vectorS.get(index) * mn);

            for (int i = 0; i < w.numRows; ++i) {
                double value = -Math.min(matrixU.get(i, index), 0.0);
                value /= uNegativeNorm;
                w.set(i, index, sqrtS * value);
            }

            for (int i = 0; i < h.numCols; ++i) {
                double value = -Math.min(matrixV.get(i, index), 0.0);
                value /= vNegativeNorm;
                h.set(index, i, sqrtS * value);
            }
        }
    }

    /**
     * Calculates l2-norm of positive elements of a column
     * @param m matrix
     * @param col index of a column
     * @return l2-norm of positive elements of the vector
     */
    private double columnPositiveNorm2(@Nonnull DMatrixRMaj m, int col) {
        double sum = 0.0;
        for (int i = 0; i < m.numRows; ++i) {
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
    private double columnNegativeNorm2(@Nonnull DMatrixRMaj m, int col) {
        double sum = 0.0;
        for (int i = 0; i < m.numRows; ++i) {
            double v = m.get(i, col);
            if (v < 0.0)
                sum += v * v;
        }
        return Math.sqrt(sum);
    }
}
