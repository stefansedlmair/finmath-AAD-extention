/**
 * 
 */
package net.finmath.montecarlo.automaticdifferentiation.backward.alternative;

import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;

/**
 * @author Stefan Sedlmair
 *
 */
public class RandomVariableAADFactory extends AbstractRandomVariableDifferentiableFactory {

	private AbstractRandomVariableFactory nonDifferentiableRandomVariableFactory = new RandomVariableFactory();
	
	/**
	 * 
	 */
	public RandomVariableAADFactory(AbstractRandomVariableFactory nonDifferentiableRandomVariableFactory) {
		this.nonDifferentiableRandomVariableFactory = nonDifferentiableRandomVariableFactory;
	}

	
	
	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.AbstractRandomVariableFactory#createRandomVariable(double, double)
	 */
	@Override
	public RandomVariableDifferentiableInterface createRandomVariable(double time, double value) {
		return new RandomVariableAAD(nonDifferentiableRandomVariableFactory, time, value);
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.AbstractRandomVariableFactory#createRandomVariable(double, double[])
	 */
	@Override
	public RandomVariableDifferentiableInterface createRandomVariable(double time, double[] values) {
		return new RandomVariableAAD(nonDifferentiableRandomVariableFactory, time, values);
	}

}
