package org.dulab.javanmf.updaterules;

import org.dulab.javanmf.algorithms.MatrixFactorization;
import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

public class CDUpdateRuleTest {

    @Test
    public void test() {

        double[][] componentArray = new double[][] {
                new double[] {0.0, 0.0},
                new double[] {0.0, 1.0},
                new double[] {0.0, 2.0},
                new double[] {1.0, 1.0},
                new double[] {2.0, 0.0},
                new double[] {1.0, 0.0},
                new double[] {0.0, 0.0}
        };

        double[][] coefficientArray = new double[][] {
                new double[] {0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0},
                new double[] {1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0}
        };

        MatrixFactorization factorization = new MatrixFactorization(
                new CDUpdateRule(0.0, 0.0),
                new CDUpdateRule(0.0, 0.0),
                1e-12, 1000);

        DoubleMatrix matrixX = new DoubleMatrix(componentArray).mmul(new DoubleMatrix(coefficientArray));
        DoubleMatrix matrixW = DoubleMatrix.rand(matrixX.rows, 2);
        DoubleMatrix matrixH = DoubleMatrix.rand(2, matrixX.columns);

        factorization.execute(matrixX, matrixW, matrixH, true);

        // Switch order of the constructed components if necessary
        int[] componentArgMaxs = matrixW.columnArgmaxs();
        if (componentArgMaxs[0] < componentArgMaxs[1]) {
            int[] order = new int[] {1, 0};
            matrixW = matrixX.getColumns(order);
            matrixH = matrixH.getRows(order);
        }

        // Scale the constructed components
        matrixW.diviRowVector(matrixW.columnMaxs().divi(2.0));
        matrixH.diviColumnVector(matrixH.rowMaxs().divi(2.0));

        Assert.assertArrayEquals(
                new DoubleMatrix(componentArray).getColumn(0).data,
                matrixW.getColumn(0).data,
                1e-6);

        Assert.assertArrayEquals(
                new DoubleMatrix(componentArray).getColumn(1).data,
                matrixW.getColumn(1).data,
                1e-6);

        Assert.assertArrayEquals(
                new DoubleMatrix(coefficientArray).getRow(0).data,
                matrixH.getRow(0).data,
                1e-6);

        Assert.assertArrayEquals(
                new DoubleMatrix(coefficientArray).getRow(1).data,
                matrixH.getRow(1).data,
                1e-6);
    }

}