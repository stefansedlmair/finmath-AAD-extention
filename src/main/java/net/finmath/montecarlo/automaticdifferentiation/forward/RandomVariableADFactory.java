/**
 * 
 */
package net.finmath.montecarlo.automaticdifferentiation.forward;

import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * @author mdm33ee
 *
 */
public class RandomVariableADFactory extends AbstractRandomVariableFactory {

	private final boolean useMultiThreading;

	public RandomVariableADFactory(boolean useMultiThreading) {
		this.useMultiThreading = useMultiThreading;
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.AbstractRandomVariableFactory#createRandomVariable(double, double)
	 */
	@Override
	public RandomVariableInterface createRandomVariable(double time, double value) {
		RandomVaribaleAD.useMultiThreading(useMultiThreading);
		return new RandomVaribaleAD(time, value);
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.AbstractRandomVariableFactory#createRandomVariable(double, double[])
	 */
	@Override
	public RandomVariableInterface createRandomVariable(double time, double[] values) {
		RandomVaribaleAD.useMultiThreading(useMultiThreading);
		return new RandomVaribaleAD(time, values);
	}

}
