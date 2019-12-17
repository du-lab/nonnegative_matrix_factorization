package org.dulab.javanmf.algorithms;

import org.ejml.data.DMatrixRMaj;
import org.junit.Test;

import static org.junit.Assert.*;
import static org.ejml.dense.row.CommonOps_DDRM.*;

public class SingularValueDecompositionTest {

    @Test
    public void decompose() {

        DMatrixRMaj expectedA = new DMatrixRMaj(new double[][] {
                new double[]{2.0, 4.0},
                new double[]{1.0, 3.0}
        });

        SingularValueDecomposition svd = new SingularValueDecomposition(expectedA);

        DMatrixRMaj w = new DMatrixRMaj(expectedA.numRows, expectedA.numRows);
        DMatrixRMaj h = new DMatrixRMaj(expectedA.numCols, expectedA.numCols);
        svd.decompose(w, h);

        DMatrixRMaj a = new DMatrixRMaj(expectedA.numRows, expectedA.numCols);
        mult(w, h, a);

        assertArrayEquals(expectedA.data, a.data, 0.3);
    }
}