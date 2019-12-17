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

import java.util.function.Function;

/**
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public class MatrixUtils {

//    public static DoubleMatrix multiply(DoubleMatrix x, DoubleMatrix y, DoubleMatrix buffer) {
//
//        if (x.rows != buffer.rows || y.columns != buffer.columns)
//            buffer.resize(x.rows, y.columns);
//
//        // Multiply x by y and assign the result to buffer
//        x.mmuli(y, buffer);
//
//        return buffer;
//    }
//
//    public static DoubleMatrix transpose(DoubleMatrix x, DoubleMatrix buffer) {
//
//        if (x.rows != buffer.columns || x.columns != buffer.rows)
//            buffer.resize(x.columns, x.rows);
//
//        for (int i = 0; i < x.rows; ++i)
//            for (int j = 0; j < x.columns; ++j)
//                buffer.put(j, i, x.get(i, j));
//
//        return buffer;
//    }

    public static void minimumEquals(DMatrixRMaj a, DMatrixRMaj b) {
        for (int i = 0; i < a.numRows; ++i)
            for (int j = 0; j < a.numCols; ++j)
                a.unsafe_set(i, j, Math.min(a.unsafe_get(i, j), b.get(i, j)));
    }
}
