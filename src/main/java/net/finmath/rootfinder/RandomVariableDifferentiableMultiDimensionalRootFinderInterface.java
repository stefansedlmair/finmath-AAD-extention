/**
 * 
 */
package net.finmath.rootfinder;

import java.util.Map;

import net.finmath.stochastic.RandomVariableInterface;

/**
 * @author Stefan Sedlmair
 *
 */
public interface RandomVariableDifferentiableMultiDimensionalRootFinderInterface {

	/**
	 * @return Next point for which a value should be set using <code>setValue</code>.
	 */
	Map<Long, RandomVariableInterface> getNextParameters();
	
	/**
	 * @param value The value corresponding to the point returned by previous <code>getNextPoint</code> call.
	 * @param derivative The derivative corresponding to the point returned by previous <code>getNextPoint</code> call.
	 */
	void setValueAndDerivative(RandomVariableInterface currentFunctionValue, Map<Long, RandomVariableInterface> gradient);

	/**
	 * @return Returns the numberOfIterations.
	 */
    int getNumberOfIterations();
	
	/**
	 * @return Returns the accuracy.
	 */
    double getAccuracy();
	
	/**
	 * @return Returns the isDone.
	 */
    boolean isDone();

	/**
	 * @return Best point optained so far
	 */
    Map<Long, RandomVariableInterface> getBestParameters();
}

