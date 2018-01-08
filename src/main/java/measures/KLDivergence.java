package measures;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import javax.annotation.Nonnull;

/**
 * Implements {@link Measure} with the Kullback-Leibler divergence
 * @author Du-Lab Team <dulab.binf@gmail.com>
 */
public class KLDivergence extends Measure
{
    /**
     * @inheritDoc
     */
    @Override
    public double get(@Nonnull DoubleMatrix x, @Nonnull DoubleMatrix w, @Nonnull DoubleMatrix h) {
        return x.mul(MatrixFunctions.log(x.div(w.mmul(h).max(1e-12)).add(x.eq(0.0)))).sub(x).add(w.mmul(h)).sum();
    }
}
