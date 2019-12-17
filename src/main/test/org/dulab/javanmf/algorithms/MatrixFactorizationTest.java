package org.dulab.javanmf.algorithms;

import org.dulab.javanmf.updaterules.MUpdateRule;
import org.ejml.data.DMatrixRMaj;
import org.junit.Test;

import java.util.Random;

import static org.ejml.dense.row.CommonOps_DDRM.*;
import static org.ejml.dense.row.RandomMatrices_DDRM.rectangle;

import static org.junit.Assert.*;

public class MatrixFactorizationTest {

    private static final double EPS = 1e-3;

    @Test
    public void test() {

        DMatrixRMaj expectedW = new DMatrixRMaj(new double[][]{
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

        MatrixFactorization factorization = new MatrixFactorization(
                new MUpdateRule(0.0, 0.0),
                new MUpdateRule(0.0, 0.0),
                1e-12, 40000);

        DMatrixRMaj matrixX = new DMatrixRMaj(expectedW.numRows, expectedH.numCols);
        mult(expectedW, expectedH, matrixX);

        Random random = new Random(3);
        DMatrixRMaj matrixW = rectangle(expectedW.numRows, expectedW.numCols, 0.1, 0.9, random);
        DMatrixRMaj matrixH = rectangle(expectedH.numRows, expectedH.numCols, 0.1, 0.9, random);

        factorization.execute(matrixX, matrixW, matrixH, true);

        // Scale columns of matrixW
        DMatrixRMaj columnFactors = new DMatrixRMaj(1, matrixW.numCols);
        maxCols(matrixW, columnFactors);
        divideCols(matrixW, columnFactors.data);
        scale(2.0, matrixW);

        // Scale rows of matrixH
        DMatrixRMaj rowFactors = new DMatrixRMaj(matrixH.numRows, 1);
        maxRows(matrixH, rowFactors);
        divideRows(rowFactors.data, matrixH);
        scale(2.0, matrixH);

        for (int i = 0; i < matrixW.numRows; ++i)
            assertEquals(expectedW.get(i, 0), matrixW.get(i, 0), EPS);

        for (int i = 0; i < matrixW.numRows; ++i)
            assertEquals(expectedW.get(i, 1), matrixW.get(i, 1), EPS);

        for (int j = 0; j < matrixH.numCols; ++j)
            assertEquals(expectedH.get(0, j), matrixH.get(0, j), EPS);

        for (int j = 0; j < matrixH.numCols; ++j)
            assertEquals(expectedH.get(1, j), matrixH.get(1, j), EPS);
    }

}