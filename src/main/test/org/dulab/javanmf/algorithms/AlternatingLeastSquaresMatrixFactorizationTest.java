package org.dulab.javanmf.algorithms;

import org.ejml.data.DMatrixRMaj;
import org.junit.Test;

import java.util.Random;

import static org.ejml.dense.row.CommonOps_DDRM.*;
import static org.ejml.dense.row.NormOps_DDRM.normF;
import static org.ejml.dense.row.RandomMatrices_DDRM.rectangle;
import static org.junit.Assert.assertEquals;

public class AlternatingLeastSquaresMatrixFactorizationTest {

    private static final double EPS = 1e-3;

    @Test
    public void test() {

        DMatrixRMaj matrixX = new DMatrixRMaj(new double[][]{
                new double[]{14.5, 2.5, 2.5, 14.5},
                new double[]{6.5, 10.5, 10.5, 6.5},
                new double[]{10.5, 6.5, 6.5, 10.5},
                new double[]{2.5, 14.5, 14.5, 2.5}
        });

        DMatrixRMaj expectedW = new DMatrixRMaj(new double[][]{
                new double[]{3.5280, 20.5039},
                new double[]{14.8461, 9.1818},
                new double[]{9.1871, 14.8428},
                new double[]{20.5047, 3.5214}
        });

        DMatrixRMaj expectedH = new DMatrixRMaj(new double[][]{
                new double[]{0.0004, 0.7071, 0.7071, 0.0005},
                new double[]{0.7072, 0.0002, 0.0003, 0.7070}
        });

        AlternatingLeastSquaresMatrixFactorization factorization = new AlternatingLeastSquaresMatrixFactorization(
                1e-12, 40000);

        Random random = new Random(0);
        DMatrixRMaj matrixW = rectangle(expectedW.numRows, expectedW.numCols, 0.0, 1.0, random);
        DMatrixRMaj matrixH = rectangle(expectedH.numRows, expectedH.numCols, 0.0, 1.0, random);

        factorization.solve(matrixX, matrixW, matrixH, true);

        DMatrixRMaj matrixE = matrixX.copy();
        multAdd(-1.0, matrixW, matrixH, matrixE);
        double error = normF(matrixE);

        assertEquals(0.0, error, 0.1);
    }

}