/**
 * 
 */
package net.finmath.montecarlo.interestrate.modelplugins;

import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretizationInterface;

/**
 * @author Stefan Sedlmair
 * 
 * @version 0.1
 * 
 * @see LIBORCorrelationModelExponentialDecay
 * @see RandomVariableDifferentiableInterface
 */
public class LIBORCorrelationModelExponentialDecay extends LIBORCorrelationModel {

	private final int numberOfFactors;
	private RandomVariableInterface parameter;
	
	/**
	 * @param timeDiscretization
	 * @param liborPeriodDiscretization
	 */
	public LIBORCorrelationModelExponentialDecay(TimeDiscretizationInterface timeDiscretization,
			TimeDiscretizationInterface liborPeriodDiscretization, int numberOfFactors, RandomVariableInterface parameter) {
		super(timeDiscretization, liborPeriodDiscretization);
		this.parameter = parameter;
		this.numberOfFactors = numberOfFactors;
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.interestrate.modelplugins.LIBORCorrelationModel#getNumberOfFactors()
	 */
	@Override
	public int getNumberOfFactors() {
		return numberOfFactors;
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.interestrate.modelplugins.LIBORCorrelationModel#clone()
	 */
	@Override
	public Object clone() {
		return new LIBORCorrelationModelExponentialDecay(timeDiscretization, liborPeriodDiscretization,  numberOfFactors, parameter);
	}

	@Override
	public void setParameter(RandomVariableInterface[] parameter) {
		if(parameter != null) this.parameter = parameter[0].abs();
	}

	@Override
	public RandomVariableInterface[] getParameter() {
		return new RandomVariableInterface[]{parameter};
	}

	
	/**
	 * Returns the entry of the factor loadings matrix which is of size #factors times #LiborPeriods
	 * 
	 *  @param timeIndex no used here
	 *  @param factorIndex index of current factor
	 *  @param liborPeriodIndex index of current LIBOR Period
	 *  
	 *  @return correlation of LIBOR at realtive factorIndex and the LIBOR at time of liborPeriodIndex
	 * */
	@Override
	public RandomVariableInterface getFactorLoading(int timeIndex, int factorIndex, int liborPeriodIndex) {
		
		double time1 = getTimeForFactorIndex(factorIndex);
		double time2 = liborPeriodDiscretization.getTime(liborPeriodIndex);
		
		return valueOnCorrleationSurface(time1, time2);
	}

	@Override
	public RandomVariableInterface getCorrelation(int timeIndex, int component1, int component2) {
		
		if(component1 > liborPeriodDiscretization.getNumberOfTimeSteps() || component2 > liborPeriodDiscretization.getNumberOfTimeSteps())
			throw new IllegalArgumentException("Libor Discretization does not support a time index of " + Math.max(component1, component2) + "!");
		
		double time1 = liborPeriodDiscretization.getTime(component1);
		double time2 = liborPeriodDiscretization.getTime(component2);
		return valueOnCorrleationSurface(time1, time2);
	}
	
	/**
	 * Volatility Surface: Corr(t<sub>1</sub>,t<sub>2</sub>) := exp(-a|t<sub>1</sub>-t<sub>2</sub>|)
	 * 
	 * @param time1 t<sub>1</sub>
	 * @param time2 t<sub>2</sub>
	 * @return correlation Corr(t<sub>1</sub>,t<sub>2</sub>) for times t<sub>1</sub> and t<sub>2</sub> 
	 * */
	private RandomVariableInterface valueOnCorrleationSurface(double time1, double time2){
		return parameter.mult(Math.abs(time1 - time2)).exp();
	}

	private double getTimeForFactorIndex(int factorIndex){
		
		double relativeFactorIndex = (double)liborPeriodDiscretization.getNumberOfTimeSteps()/(double)getNumberOfFactors() * (double)factorIndex;
		
		double tplus 	= liborPeriodDiscretization.getTime((int) Math.ceil(relativeFactorIndex));
		double tminus 	= liborPeriodDiscretization.getTime((int) Math.floor(relativeFactorIndex));

		return tminus + (relativeFactorIndex%1) * (tplus - tminus);
	}
	
}
