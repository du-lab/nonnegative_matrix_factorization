package org.dulab.javanmf.algorithms;

import org.ejml.data.DMatrixRMaj;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.ejml.dense.row.CommonOps_DDRM.*;


public class BroJongNonNegativeLeastSquares {

    private static final double TOLERANCE = 1e-4;

    public void solve(DMatrixRMaj matrixX, DMatrixRMaj matrixZ, DMatrixRMaj matrixD) {

        if (matrixX.numCols != matrixD.numCols || matrixZ.numCols != matrixD.numRows || matrixX.numRows != matrixZ.numRows)
            throw new IllegalStateException("Wrong shape of the input matrices");

        // Initialize variables
        List<Set<Integer>> passiveSets = new ArrayList<>(matrixD.numCols);
        List<Set<Integer>> activeSets = new ArrayList<>(matrixD.numCols);
        for (int i = 0; i < matrixD.numCols; ++i) {
            passiveSets.add(new HashSet<>());
            Set<Integer> activeSet = new HashSet<>();
            for (int j = 0; j < matrixZ.numCols; ++j)
                activeSet.add(j);
            activeSets.add(activeSet);
        }

        DMatrixRMaj matrixZtX = new DMatrixRMaj(matrixZ.numCols, matrixX.numCols);
        multTransA(matrixZ, matrixX, matrixZtX);

        DMatrixRMaj matrixZtZ = new DMatrixRMaj(matrixZ.numCols, matrixZ.numCols);
        multInner(matrixZ, matrixZtZ);

        DMatrixRMaj matrixW = calculateMatrixW(matrixZtX, matrixZtZ, matrixD);
        // Main loop
        int[] maximumIndices = findActiveMaximumIndices(matrixW, activeSets, null);
        while(!checkAllEmpty(activeSets) && findMaximum(matrixW, maximumIndices) > TOLERANCE) {

            for (int column = 0; column < matrixD.numCols; ++column) {

                Set<Integer> activeSet = activeSets.get(column);
                Set<Integer> passiveSet = passiveSets.get(column);
                int m = maximumIndices[column];

                if (activeSet.isEmpty() || matrixW.unsafe_get(m, column) <= TOLERANCE)
                    continue;

                passiveSet.add(m);
                activeSet.remove(m);
                DMatrixRMaj vectorS = calculateVectorS(matrixZtZ, matrixZtX, column, passiveSet);

                // Inner loop
                while (findMinimum(vectorS, passiveSet) <= 0) {
                    updateMatrixD(matrixD, column, vectorS, passiveSet);
                    updateSets(matrixD, column, passiveSet, activeSet);
                    vectorS = calculateVectorS(matrixZtZ, matrixZtX, column, passiveSet);
                }

                for (int i = 0; i < matrixD.numRows; ++i)
                    matrixD.unsafe_set(i, column, vectorS.unsafe_get(i, 0));
            }

            matrixW = calculateMatrixW(matrixZtX, matrixZtZ, matrixD);
            maximumIndices = findActiveMaximumIndices(matrixW, activeSets, maximumIndices);
        }

    }

    /**
     * Calculates W = ZtX - ZtZ x D
     *
     * @param matrixZtX matrix ZtX
     * @param matrixZtZ matrix ZtZ
     * @param matrixD   matrix D
     * @return matrix W
     */
    private DMatrixRMaj calculateMatrixW(DMatrixRMaj matrixZtX, DMatrixRMaj matrixZtZ, DMatrixRMaj matrixD) {
        DMatrixRMaj matrixW = matrixZtX.copy();
        multAdd(-1.0, matrixZtZ, matrixD, matrixW);
        return matrixW;
    }

    /**
     * Finds indices of the maximums in each column of matrix W among active rows
     *
     * @param matrixW    matrix W
     * @param activeSets list of active sets
     * @param buffer     buffer for calculated indices
     * @return indices
     */
    private int[] findActiveMaximumIndices(DMatrixRMaj matrixW, List<Set<Integer>> activeSets, int[] buffer) {

        if (buffer == null)
            buffer = new int[matrixW.numCols];

        for (int j = 0; j < matrixW.numCols; ++j) {
            Set<Integer> activeSet = activeSets.get(j);
            double maximum = -Double.MAX_VALUE;
            int maximumIndex = -1;
            for (int rowIndex : activeSet) {
                double w = matrixW.unsafe_get(rowIndex, j);
                if (w > maximum) {
                    maximum = w;
                    maximumIndex = rowIndex;
                }
            }
            buffer[j] = maximumIndex;
        }

        return buffer;
    }

    private double findMaximum(DMatrixRMaj matrixW, int[] rows) {
        double maximum = -Double.MAX_VALUE;
        for (int i = 0; i < matrixW.numCols; ++i) {
            double x = matrixW.unsafe_get(rows[i], i);
            if (x > maximum)
                maximum = x;
        }
        return maximum;
    }

    private double findMinimum(DMatrixRMaj vectorS, Set<Integer> passiveSet) {
        double minimum = Double.MAX_VALUE;
        for (int i : passiveSet) {
            double x = vectorS.get(i);
            if (x < minimum)
                minimum = x;
        }
        return minimum;
    }

    /**
     * Calculates alpha = -min(d / (d - s)) over indices from the passive set
     * @param matrixD matrix D
     * @param column index of a column in matrix D
     * @param vectorS vector S
     * @param passiveSet passive set of indices
     * @return alpha
     */
    private double calculateAlpha(DMatrixRMaj matrixD, int column, DMatrixRMaj vectorS, Set<Integer> passiveSet) {
        double minimum = Double.MAX_VALUE;
        for (int i : passiveSet) {

            double s = vectorS.get(i);
            if (s > 0.0)
                continue;

            double x = matrixD.unsafe_get(i, column);
            x /= x - s;
            if (x < minimum)
                minimum = x;
        }
        return minimum;
    }

    /**
     * Returns true if all sets are empty. Otherwise, returns false.
     * @param sets list of sets
     * @return true if all sets are empty
     */
    private boolean checkAllEmpty(List<Set<Integer>> sets) {
        for (Set<Integer> set : sets)
            if (!set.isEmpty()) return false;
        return true;
    }

    /**
     * Calculates S, where SP = [(ZtZ)P]^(-1) x (ZtX)P and SR = 0
     *
     * @param matrixZtZ   matrix ZtZ
     * @param matrixZtX   matrix ZtX
     * @param columnIndex column of ZtX
     * @param passiveSet  passive set
     * @return vector SP
     */
    private DMatrixRMaj calculateVectorS(DMatrixRMaj matrixZtZ, DMatrixRMaj matrixZtX, int columnIndex, Set<Integer> passiveSet) {

        DMatrixRMaj matrixZtZP = new DMatrixRMaj(passiveSet.size(), passiveSet.size());
        int ii = 0;
        for (int i : passiveSet) {
            int jj = 0;
            for (int j : passiveSet) {
                matrixZtZP.unsafe_set(ii, jj++, matrixZtZ.unsafe_get(i, j));
            }
            ++ii;
        }

        DMatrixRMaj vectorZtXP = new DMatrixRMaj(passiveSet.size(), 1);
        ii = 0;
        for (int i : passiveSet)
            vectorZtXP.unsafe_set(ii++, 0, matrixZtX.unsafe_get(i, columnIndex));

        if (!invert(matrixZtZP))
            throw new IllegalStateException("Cannot invert matrix ZtZP");

        DMatrixRMaj vectorSP = new DMatrixRMaj(passiveSet.size(), 1);
        mult(matrixZtZP, vectorZtXP, vectorSP);

        DMatrixRMaj vectorS = new DMatrixRMaj(matrixZtX.numRows, 1);
        ii = 0;
        for (int i : passiveSet)
            vectorS.unsafe_set(i, 0, vectorSP.unsafe_get(ii++, 0));

        return vectorS;
    }

    /**
     * Update matrix D s.t. D <- D + alpha (S - D)
     * @param matrixD matrix D
     * @param column index of a column of matrix D
     * @param vectorS vector S
     * @param passiveSet passive set of indices
     */
    private void updateMatrixD(DMatrixRMaj matrixD, int column, DMatrixRMaj vectorS, Set<Integer> passiveSet) {

        double alpha = calculateAlpha(matrixD, column, vectorS, passiveSet);

        for (int i = 0; i < matrixD.numRows; ++i) {
            double x = matrixD.unsafe_get(i, column);
            x += alpha * (vectorS.get(i) - x);
            matrixD.unsafe_set(i, column, x);
        }
    }

    /**
     * If some values of D are close to zero, their indices are removed from the passive set and added to the active set
     * @param matrixD matrix D
     * @param column index of a column of matrix D
     * @param passiveSet set of passive indices
     * @param activeSet set of active indices
     */
    private void updateSets(DMatrixRMaj matrixD, int column, Set<Integer> passiveSet, Set<Integer> activeSet) {
        for (int i = 0; i < matrixD.numRows; ++i) {
            double d = matrixD.unsafe_get(i, column);
            if (d == 0.0) {
                passiveSet.remove(i);
                activeSet.add(i);
            }
        }
    }
}
