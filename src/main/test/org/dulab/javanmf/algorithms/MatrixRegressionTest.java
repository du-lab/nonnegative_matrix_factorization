package org.dulab.javanmf.algorithms;

import org.dulab.javanmf.updaterules.MUpdateRule;
import org.ejml.data.DMatrixRMaj;
import org.junit.Test;

import java.util.Random;

import static org.ejml.dense.row.CommonOps_DDRM.*;
import static org.ejml.dense.row.RandomMatrices_DDRM.rectangle;
import static org.junit.Assert.assertEquals;

public class MatrixRegressionTest {

    private static final double EPS = 1e-5;

    @Test
    public void test() {

        DMatrixRMaj matrixW = new DMatrixRMaj(new double[][]{
                new double[]{0.0, 0.0},
                new double[]{0.0, 1.0},
                new double[]{0.0, 2.0},
                new double[]{1.0, 1.0},
                new double[]{2.0, 0.0},
                new double[]{1.0, 0.0},
                new double[]{0.0, 0.0}
        });

        DMatrixRMaj expectedH = new DMatrixRMaj(new double[][]{
                new double[]{0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0},
                new double[]{1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0}
        });

        MatrixRegression regression = new MatrixRegression(
                new MUpdateRule(0.0, 0.0),
                1e-12, 40000);

        DMatrixRMaj matrixX = new DMatrixRMaj(matrixW.numRows, expectedH.numCols);
        mult(matrixW, expectedH, matrixX);

        Random random = new Random(3);
        DMatrixRMaj matrixH = rectangle(expectedH.numRows, expectedH.numCols, 0.1, 0.9, random);

        regression.solve(matrixX, matrixW, matrixH, true);

        for (int j = 0; j < matrixH.numCols; ++j)
            assertEquals(expectedH.get(0, j), matrixH.get(0, j), EPS);

        for (int j = 0; j < matrixH.numCols; ++j)
            assertEquals(expectedH.get(1, j), matrixH.get(1, j), EPS);
    }

}