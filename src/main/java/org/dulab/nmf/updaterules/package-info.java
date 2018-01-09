/**
 * Provides classes for updating matrix H in the direction of minimising the distance D(X, WH).
 *
 * <ul>
 *     <li>Multiplicative update rule for the euclidean distance with regularization</li>
 *     <li>Multiplicative update tule for the Kullback-Leibler divergence with regularization</li>
 *     <li>Fast-gradient-descent update rule for the euclidean distance with regularization</li>
 *     <li>Fast-gradient-descent update rule for the Kullback-Leibler divergence with regularization</li>
 * </ul>
 *
 * @see org.dulab.nmf.updaterules.MUpdateRule
 * @see org.dulab.nmf.updaterules.MKLUpdateRule
 * @see org.dulab.nmf.updaterules.FGDMUpdateRule
 * @see org.dulab.nmf.updaterules.FGDKLUpdateRule
 * @author Du-Lab Team dulab.binf@gmail.com
 */
package org.dulab.nmf.updaterules;