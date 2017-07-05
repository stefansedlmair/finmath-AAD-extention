/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christianfries.com.
 *
 * Created on 17.06.2017
 */

package net.finmath.montecarlo.automaticdifferentiation;

import java.util.Map;

import net.finmath.stochastic.RandomVariableInterface;

/**
 * Interface providing additional methods for
 * random variable implementing <code>RandomVariableInterface</code>
 * allowing automatic differentiation.
 * 
 * The interface will introduce two additional methods: <code>Long getID()</code> and
 * <long>Map<Long, RandomVariableInterface> getGradient()</code>.
 * The method <code>getGradient</code> will return a map providing the first order
 * differentiation of the given random variable (this) with respect to
 * <i>all</i> its input <code>RandomVariableDifferentiableInterface</code>s (leave nodes).
 * 
 * To get the differentiation with respect to a specific object use
 * <code>
 * <pre>
 * 		Map gradient = X.getGradient();
 * 		RandomVariableInterface derivative = X.get(Y.getID());
 * </pre>
 * </code>
 * 
 * @author Christian Fries
 */
public interface RandomVariableDifferentiableInterface extends RandomVariableInterface {
	
	/**
	 * A unique id for this random variable. Will be used in <code>getGradient</code>.
	 * 
	 * @return The id for this random variable.
	 */
	Long getID();
	
	/**
	 * Returns the gradient of this random variable with respect to all its leaf nodes.
	 * The method calculated the map \( v \mapsto \frac{d u}{d v} \) where \( u \) denotes <code>this</code>.
	 * 
	 * @return The gradient map.
	 */
	Map<Long, RandomVariableInterface> getGradient();
	
}
