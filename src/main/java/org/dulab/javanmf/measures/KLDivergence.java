package org.dulab.javanmf.measures;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.ojalgo.matrix.PrimitiveMatrix;
import org.ojalgo.matrix.store.PrimitiveDenseStore;

import javax.annotation.Nonnull;

/**
 * Calculates the distance between matrices X and WH using the generalized Kullback-Leibler divergence<br>
 * &sum; ( X log ( X / WH ) &minus; X + WH )
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public class KLDivergence extends Measure
{
    @Override
    public double get(@Nonnull PrimitiveMatrix x,
                      @Nonnull PrimitiveMatrix w,
                      @Nonnull PrimitiveMatrix h) {


         (w.multiply(h).add(1e-12))

        return x.mul(MatrixFunctions.log(x.div(w.mmul(h).max(1e-12)).add(x.eq(0.0)))).sub(x).add(w.mmul(h)).sum();
    }
}
