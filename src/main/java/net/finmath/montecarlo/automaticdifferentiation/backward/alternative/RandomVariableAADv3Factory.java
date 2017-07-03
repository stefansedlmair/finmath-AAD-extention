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
public class RandomVariableAADv3Factory extends AbstractRandomVariableFactory {

	/**
	 * 
	 */
	public RandomVariableAADv3Factory() {
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.AbstractRandomVariableFactory#createRandomVariable(double, double)
	 */
	@Override
	public RandomVariableInterface createRandomVariable(double time, double value) {
		return new RandomVariableAADv3(time, value);
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.AbstractRandomVariableFactory#createRandomVariable(double, double[])
	 */
	@Override
	public RandomVariableInterface createRandomVariable(double time, double[] values) {
		return new RandomVariableAADv3(time, values);
	}

}
