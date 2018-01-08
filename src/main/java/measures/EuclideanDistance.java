package measures;

import org.jblas.DoubleMatrix;

import javax.annotation.Nonnull;

/**
 * Implements {@link Measure} with the euclidean distance: ||X - W x H||^2
 * @author Du-Lab Team <dulab.binf@gmail.com>
 */
public class EuclideanDistance extends Measure
{
    /**
     * @inheritDoc
     */
    @Override
    public double get(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h) {
        double norm = x.sub(w.mmul(h)).norm2();
        return norm * norm;
    }
}
