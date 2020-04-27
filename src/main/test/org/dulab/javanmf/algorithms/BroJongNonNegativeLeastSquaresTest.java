package org.dulab.javanmf.algorithms;

import org.ejml.data.DMatrixRMaj;
import org.junit.Test;

import static org.junit.Assert.*;

public class BroJongNonNegativeLeastSquaresTest {

    private static final double EPS = 1e-3;

    @Test
    public void solve() {

        DMatrixRMaj matrixZ = new DMatrixRMaj(new double[][]{
                new double[]{73, 71, 52},
                new double[]{87, 74, 46},
                new double[]{72, 2, 7},
                new double[]{80, 89, 71}
        });

        DMatrixRMaj matrixX = new DMatrixRMaj(new double[][]{
                new double[]{49, 98},
                new double[]{67, 134},
                new double[]{68, 136},
                new double[]{20, 40}
        });

        DMatrixRMaj expectedD = new DMatrixRMaj(new double[][]{
                new double[]{0.65, 1.3},
                new double[]{0.0, 0.0},
                new double[]{0.0, 0.0}
        });

        DMatrixRMaj matrixD = new DMatrixRMaj(matrixZ.numCols, matrixX.numCols);

        new BroJongNonNegativeLeastSquares().solve(matrixX, matrixZ, matrixD);

        assertArrayEquals(expectedD.data, matrixD.data, EPS);
    }

    @Test
    public void solve2() {

        DMatrixRMaj matrixZ = new DMatrixRMaj(new double[][]{
                new double[]{0.0372, 0.2869},
                new double[]{0.6861, 0.7071},
                new double[]{70.6233, 0.6245},
                new double[]{0.6344, 0.6170}
        });

        DMatrixRMaj matrixX = new DMatrixRMaj(new double[]{0.8587, 0.1781, 0.0747, 0.8405});

        DMatrixRMaj expectedD = new DMatrixRMaj(new double[]{0, 0.6929});

        DMatrixRMaj matrixD = new DMatrixRMaj(matrixZ.numCols, matrixX.numCols);

        new BroJongNonNegativeLeastSquares().solve(matrixX, matrixZ, matrixD);

        assertArrayEquals(expectedD.data, matrixD.data, EPS);
    }
}