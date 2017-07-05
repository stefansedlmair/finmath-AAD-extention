/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 20.05.2006
 */
package net.finmath.montecarlo.interestrate.modelplugins;

import net.finmath.montecarlo.RandomVariable;
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

	private RandomVariableInterface[] parameter = new RandomVariableInterface[5];

	private LIBORVolatilityModel	volatilityModel;
	private LIBORCorrelationModel	correlationModel;
	
	public LIBORCovarianceModelExponentialForm5Param(TimeDiscretizationInterface timeDiscretization, TimeDiscretizationInterface liborPeriodDiscretization, int numberOfFactors, RandomVariableInterface[] parameters) {
		super(timeDiscretization, liborPeriodDiscretization, numberOfFactors);
		
		this.parameter = parameters.clone();
		volatilityModel		= new LIBORVolatilityModelFourParameterExponentialForm(getTimeDiscretization(), getLiborPeriodDiscretization(), parameter[0], parameter[1], parameter[2], parameter[3]);
		correlationModel	= new LIBORCorrelationModelExponentialDecay(getLiborPeriodDiscretization(), getLiborPeriodDiscretization(), getNumberOfFactors(), parameter[4]);
	}

	@Override
	public Object clone() {
		LIBORCovarianceModelExponentialForm5Param model = new LIBORCovarianceModelExponentialForm5Param(this.getTimeDiscretization(), this.getLiborPeriodDiscretization(), this.getNumberOfFactors(), parameter);
		model.parameter = this.parameter;
		model.volatilityModel = this.volatilityModel;
		model.correlationModel = this.correlationModel;
		return model;
	}
	
	@Override
	public AbstractLIBORCovarianceModelParametric getCloneWithModifiedParameters(double[] parameters) {
		throw new IllegalArgumentException("Cannot acceppt doubles as Parameters!");
	}

	@Override
	public double[] getParameter() {
		throw new UnsupportedOperationException("Method has wrong return type!");
	}

	@Override
    public RandomVariableInterface[] getFactorLoading(int timeIndex, int component, RandomVariableInterface[] realizationAtTimeIndex) {
		RandomVariableInterface[] factorLoading = new RandomVariableInterface[correlationModel.getNumberOfFactors()];
		for (int factorIndex = 0; factorIndex < factorLoading.length; factorIndex++) {
			RandomVariableInterface volatility = volatilityModel.getVolatility(timeIndex, component);
			factorLoading[factorIndex] = volatility
                    .mult(correlationModel.getFactorLoading(timeIndex, factorIndex, component));
		}
		
		return factorLoading;
	}

	@Override
	public RandomVariable getFactorLoadingPseudoInverse(int timeIndex, int component, int factor, RandomVariableInterface[] realizationAtTimeIndex) {
		throw new UnsupportedOperationException();
	}
}
