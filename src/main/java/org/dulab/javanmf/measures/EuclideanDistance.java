package org.dulab.javanmf.measures;

import org.jblas.DoubleMatrix;
import org.ojalgo.matrix.store.PrimitiveDenseStore;

import javax.annotation.Nonnull;
import java.util.stream.IntStream;

/**
 * Calculates the distance between matrices X and WH using the euclidean distance || X &minus; WH ||<sup>2</sup>
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public class EuclideanDistance extends Measure
{
    @Override
    public double get(@Nonnull PrimitiveDenseStore x,
                      @Nonnull PrimitiveDenseStore w,
                      @Nonnull PrimitiveDenseStore h) {

//        double norm = x.sub(w.mmul(h)).norm2();

        // Calculate ||X - W x H||^2
        double norm2 = 0.0;
        for (int i = 0; i < x.countRows(); ++i)
            for (int j = 0; j < x.countColumns(); ++j) {
                double residual = x.get(i, j);
                for (int k = 0; k < w.countColumns(); ++k)
                    residual -= w.get(i, k) * h.get(k, j);
                norm2 += residual * residual;
            }

        return norm2;
    }
}
