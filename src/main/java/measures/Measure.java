package measures;

import org.jblas.DoubleMatrix;

import javax.annotation.Nonnull;

/**
 * This abstract class is used to calculate distance between matrices X and W x H
 * @author Du-Lab Team <dulab.binf@gmail.com>
 */
public abstract class Measure
{
    /**
     * Returns distance between matrices X and W x H
     * @param x matrix of shape [num_features, num_samples]
     * @param w matrix of shape [num_features, num_components]
     * @param h matrix of shape [num_components, num_samples
     * @return distance value
     */
    public abstract double get(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h);
}
