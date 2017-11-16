/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 15.12.2007
 */
package net.finmath.montecarlo.interestrate.modelplugins;

import java.util.Arrays;

import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretizationInterface;

/**
 * A covariance model build from a volatility model implementing
 * <code>LIBORVolatilityModel</code> and a correlation model
 * implementing <code>LIBORCorrelationModel</code>.
 * 
 * <p>
 * The model parameters are given by the concatenation of the
 * parameters of the <code>LIBORVolatilityModel</code> and
 * the parameters of the <code>LIBORCorrelationModel</code>,
 * in this ordering
 * </p>
 * 
 * @author Christian Fries
 */
public class LIBORCovarianceModelFromVolatilityAndCorrelation extends AbstractLIBORCovarianceModelParametric {

	private LIBORVolatilityModel	volatilityModel;
	private LIBORCorrelationModel	correlationModel;
	
	public LIBORCovarianceModelFromVolatilityAndCorrelation(TimeDiscretizationInterface timeDiscretization, TimeDiscretizationInterface liborPeriodDiscretization, LIBORVolatilityModel volatilityModel, LIBORCorrelationModel correlationModel) {
		super(timeDiscretization, liborPeriodDiscretization, correlationModel.getNumberOfFactors());

		this.volatilityModel = volatilityModel;
		this.correlationModel = correlationModel;
	}

	@Override
    public RandomVariableInterface[] getFactorLoading(int timeIndex, int component, RandomVariableInterface[] realizationAtTimeIndex) {
		RandomVariableInterface[] factorLoading = new RandomVariableInterface[correlationModel.getNumberOfFactors()];

		RandomVariableInterface volatility	= volatilityModel.getVolatility(timeIndex, component);
		for (int factorIndex = 0; factorIndex < factorLoading.length; factorIndex++) {
			factorLoading[factorIndex] = volatility.mult(correlationModel.getFactorLoading(timeIndex, factorIndex, component));
		}
		
		return factorLoading;
	}
	
	@Override
    public RandomVariableInterface getFactorLoadingPseudoInverse(int timeIndex, int component, int factor, RandomVariableInterface[] realizationAtTimeIndex) {
		// Note that we assume that the correlation model getFactorLoading gives orthonormal vectors
		RandomVariableInterface factorLoadingPseudoInverse = volatilityModel.getVolatility(timeIndex, component).invert()
                .mult(correlationModel.getFactorLoading(timeIndex, factor, component));

        // @todo numberOfComponents should be stored as a member?!
        int numberOfComponents = getLiborPeriodDiscretization().getNumberOfTimeSteps();
        
        RandomVariableInterface factorWeight = null;
        for(int componentIndex=0; componentIndex<numberOfComponents; componentIndex++) {
            RandomVariableInterface factorElement = correlationModel.getFactorLoading(timeIndex, factor, componentIndex);            
            factorWeight = factorWeight == null ? factorElement.squared() : factorWeight.add(factorElement.squared());                                                                                                                 
        }

        factorLoadingPseudoInverse = factorLoadingPseudoInverse.div(factorWeight);

        return factorLoadingPseudoInverse;		
	}

    /* (non-Javadoc)
     * @see net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModel#getCovariance(int, int, int)
     */
    @Override
    public RandomVariableInterface getCovariance(int timeIndex, int component1, int component2, RandomVariableInterface[] realizationAtTimeIndex) {
//        RandomVariableInterface covariance = new RandomVariable(0.0, correlationModel.getCorrelation(timeIndex, component1, component2));
//        covariance = covariance.mult(volatilityModel.getVolatility(timeIndex, component1))
//                .mult(volatilityModel.getVolatility(timeIndex, component2));
    	
    	RandomVariableInterface volatilityOfComponent1 = volatilityModel.getVolatility(timeIndex, component1);
    	RandomVariableInterface volatilityOfComponent2 = volatilityModel.getVolatility(timeIndex, component2);
    	
    	RandomVariableInterface correlationOfComponent1And2 = correlationModel.getCorrelation(timeIndex, component1, component2);
    	
    	RandomVariableInterface covariance = volatilityOfComponent1.mult(volatilityOfComponent2).mult(correlationOfComponent1And2);
    	
        return covariance;
    }
	
	@Override
	public RandomVariableInterface[] getParameterAsRandomVariable() {
		// get parameters
		RandomVariableInterface[] volatilityParameter = volatilityModel.getParameterAsRandomVariable();
		RandomVariableInterface[] correlationParameter = correlationModel.getParameterAsRandomVariable();

		// cover case of not calibrateable models
		if(volatilityParameter  == null) volatilityParameter  = new RandomVariableInterface[] {};
		if(correlationParameter == null) correlationParameter = new RandomVariableInterface[] {};

		// copy parameters into new storage
		int numberOfParameters = volatilityParameter.length + correlationParameter.length;
		
		RandomVariableInterface[] parameter = new RandomVariableInterface[numberOfParameters];
		
		System.arraycopy(volatilityParameter,  0, parameter, 0, 							volatilityParameter.length);
		System.arraycopy(correlationParameter, 0, parameter, volatilityParameter.length, 	parameter.length - volatilityParameter.length);

		return parameter;
	}
		
	@Override
	public Object clone() {
		return new LIBORCovarianceModelFromVolatilityAndCorrelation(
				this.getTimeDiscretization(),
				this.getLiborPeriodDiscretization(),
				(LIBORVolatilityModel)volatilityModel.clone(), 
				(LIBORCorrelationModel)correlationModel.clone());
	}

	@Override
	public AbstractLIBORCovarianceModelParametric getCloneWithModifiedParameters(double[] parameters) {
		LIBORVolatilityModel volatilityModel = this.volatilityModel;
		LIBORCorrelationModel correlationModel = this.correlationModel;

		double[] volatilityParameter = volatilityModel.getParameter();
		double[] correlationParameter = correlationModel.getParameter();

		int parameterIndex = 0;
		if(volatilityParameter != null) {
			double[] newVolatilityParameter = new double[volatilityParameter.length];
			System.arraycopy(parameters, parameterIndex, newVolatilityParameter, 0, newVolatilityParameter.length);
			parameterIndex += newVolatilityParameter.length;
			if(!Arrays.equals(newVolatilityParameter, volatilityModel.getParameter())) {
				volatilityModel = ((LIBORVolatilityModel) volatilityModel.clone());
				volatilityModel.setParameter(newVolatilityParameter);
			}
		}
				
		if(correlationParameter != null) {
			double[] newCorrelationParameter = new double[correlationParameter.length];
			System.arraycopy(parameters, parameterIndex, newCorrelationParameter, 0, newCorrelationParameter.length);
			parameterIndex += newCorrelationParameter.length;
			if(!Arrays.equals(newCorrelationParameter, correlationModel.getParameter()))
				correlationModel = ((LIBORCorrelationModel) correlationModel.clone());
				correlationModel.setParameter(newCorrelationParameter);
		}
		return new LIBORCovarianceModelFromVolatilityAndCorrelation(this.getTimeDiscretization(), this.getLiborPeriodDiscretization(), volatilityModel, correlationModel);
	}

	public LIBORVolatilityModel getVolatilityModel() {
		return volatilityModel;
	}

	public LIBORCorrelationModel getCorrelationModel() {
		return correlationModel;
	}


}
