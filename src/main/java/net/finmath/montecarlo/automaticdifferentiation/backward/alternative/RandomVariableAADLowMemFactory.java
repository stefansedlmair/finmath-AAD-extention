/**
 * 
 */
package net.finmath.montecarlo.automaticdifferentiation.backward.alternative;

import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * @author Stefan Sedlmair
 *
 */
public class RandomVariableAADLowMemFactory extends AbstractRandomVariableFactory {

	/**
	 * 
	 */
	public RandomVariableAADLowMemFactory() {
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.AbstractRandomVariableFactory#createRandomVariable(double, double)
	 */
	@Override
	public RandomVariableInterface createRandomVariable(double time, double value) {
		return new RandomVariableAADLowMem(time, value);
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.AbstractRandomVariableFactory#createRandomVariable(double, double[])
	 */
	@Override
	public RandomVariableInterface createRandomVariable(double time, double[] values) {
		return new RandomVariableAADLowMem(time, values);
	}

}
