package org.dulab.javanmf.measures;

import org.ejml.data.DMatrixRMaj;

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
    public abstract double get(@Nonnull DMatrixRMaj x, @Nonnull DMatrixRMaj w, @Nonnull DMatrixRMaj h);
}
