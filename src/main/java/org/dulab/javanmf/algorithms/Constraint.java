package org.dulab.javanmf.algorithms;

import org.ejml.data.DMatrixRMaj;

public interface Constraint {

    DMatrixRMaj apply(DMatrixRMaj matrix);
}
