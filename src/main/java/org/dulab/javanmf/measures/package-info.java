/**
 * Provides classes for estimating the distance between matrices X and WH.
 *
 * <ul>
 *     <li>Euclidean distance || X &minus; WH ||<sup>2</sup></li>
 *     <li>Generalized Kullback-Leibler divergence &sum; ( X log ( X / WH ) &minus; X + WH )</li>
 * </ul>
 *
 * @see org.dulab.javanmf.measures.EuclideanDistance
 * @see org.dulab.javanmf.measures.KLDivergence
 * @author Du-Lab Team dulab.binf@gmail.com
 */
package org.dulab.javanmf.measures;