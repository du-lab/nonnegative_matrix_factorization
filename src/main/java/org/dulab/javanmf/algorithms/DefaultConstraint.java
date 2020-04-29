package org.dulab.javanmf.algorithms;

import org.ejml.data.DMatrixRMaj;

public class DefaultConstraint implements Constraint {

    @Override
    public DMatrixRMaj apply(DMatrixRMaj matrix) {
        return matrix;
    }
}
