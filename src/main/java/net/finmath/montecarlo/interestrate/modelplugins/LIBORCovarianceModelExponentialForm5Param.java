/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 20.05.2006
 */
package net.finmath.montecarlo.interestrate.modelplugins;

import java.util.stream.IntStream;

import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.RandomVariable;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretizationInterface;

/**
 * The five parameter covariance model consisting of an
 * {@link LIBORVolatilityModelMaturityDependentFourParameterExponentialForm}
 * and an
 * {@link LIBORCorrelationModelExponentialDecay}.
 * 
 * @author Christian Fries
 */
public class LIBORCovarianceModelExponentialForm5Param extends AbstractLIBORCovarianceModelParametric {

	final private LIBORVolatilityModel	volatilityModel;
	final private LIBORCorrelationModel	correlationModel;
	
	final private AbstractRandomVariableFactory randomVariableFactory;
	
	private LIBORCovarianceModelExponentialForm5Param(AbstractRandomVariableFactory randomVariableFactory, LIBORVolatilityModel	volatilityModel, LIBORCorrelationModel	correlationModel) {
		super(volatilityModel.getTimeDiscretization(), volatilityModel.getLiborPeriodDiscretization(), correlationModel.getNumberOfFactors());
		
		this.randomVariableFactory 	= randomVariableFactory; 
		this.volatilityModel		= volatilityModel;
		this.correlationModel		= correlationModel;
	}
	
	public LIBORCovarianceModelExponentialForm5Param(AbstractRandomVariableFactory randomVariableFactory, TimeDiscretizationInterface timeDiscretization, TimeDiscretizationInterface liborPeriodDiscretization, int numberOfFactors, double[] parameters) {
		this(randomVariableFactory,
				new LIBORVolatilityModelFourParameterExponentialForm(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, parameters[0], parameters[1], parameters[2], parameters[3], true),
				new LIBORCorrelationModelExponentialDecay(liborPeriodDiscretization, liborPeriodDiscretization, numberOfFactors, parameters[4], false)
				);
	}

	public LIBORCovarianceModelExponentialForm5Param(TimeDiscretizationInterface timeDiscretization, TimeDiscretizationInterface liborPeriodDiscretization, int numberOfFactors) {
		this(new RandomVariableFactory(), timeDiscretization, liborPeriodDiscretization, numberOfFactors, new double[] { 0.20, 0.05, 0.10, 0.20, 0.10});
	}
	
	@Override
	public Object clone() {
		return new LIBORCovarianceModelExponentialForm5Param(randomVariableFactory, volatilityModel, correlationModel);
	}
	
	@Override
	public AbstractLIBORCovarianceModelParametric getCloneWithModifiedParameters(double[] parameters) {	
		// clone this 
		LIBORCovarianceModelExponentialForm5Param clone = (LIBORCovarianceModelExponentialForm5Param) this.clone();
		
		// if necessary change parameters of volatility model from cloned model
		double[] volatilityParameters = clone.volatilityModel.getParameter();
		if(parameters[0] != volatilityParameters[0] || parameters[1] != volatilityParameters[1] || parameters[2] != volatilityParameters[2] || parameters[3] != volatilityParameters[3])
			clone.volatilityModel.setParameter(parameters);
		
		// correlation model not calibratable at the moment
		return clone;
	}

	@Override
    public RandomVariableInterface[] getFactorLoading(int timeIndex, int component, RandomVariableInterface[] realizationAtTimeIndex) {
		RandomVariableInterface volatility = volatilityModel.getVolatility(timeIndex, component);
		
		RandomVariableInterface[] factorLoading = IntStream.range(0, correlationModel.getNumberOfFactors())
				.mapToObj(factorIndex -> volatility.mult(correlationModel.getFactorLoading(timeIndex, factorIndex, component)))
				.toArray(RandomVariableInterface[]::new);
				
		return factorLoading;
	}

	@Override
	public RandomVariable getFactorLoadingPseudoInverse(int timeIndex, int component, int factor, RandomVariableInterface[] realizationAtTimeIndex) {
		throw new UnsupportedOperationException();
	}

	@Override
	public RandomVariableInterface[] getParameterAsRandomVariable() {
		// only Volatility model is calibrateable at this point
		return volatilityModel.getParameterAsRandomVariable();
	}
}
