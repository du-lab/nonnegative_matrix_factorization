package measures;

import org.jblas.DoubleMatrix;

import javax.annotation.Nonnull;

/**
 * Provides a template for calculating the distance between matrices X and WH
 * @author Du-Lab Team dulab.binf@gmail.com
 */
public abstract class Measure
{
    /**
     * Returns distance between matrices X and WH
     * @param x matrix of shape [N<sub>points</sub>, N<sub>vectors</sub>]
     * @param w matrix of shape [N<sub>points</sub>, N<sub>components</sub>]
     * @param h matrix of shape [N<sub>components</sub>, N<sub>vectors</sub>]
     * @return distance value
     */
    public abstract double get(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h);
}
