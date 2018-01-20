package org.dulab.javanmf.measures;

import org.jblas.DoubleMatrix;

import javax.annotation.Nonnull;

/**
 * Calculates the distance between matrices X and WH using the euclidean distance || X &minus; WH ||<sup>2</sup>
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public class EuclideanDistance extends Measure
{
    @Override
    public double get(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h) {
        double norm = x.sub(w.mmul(h)).norm2();
        return norm * norm;
    }
}
