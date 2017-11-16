/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 20.05.2006
 */
package net.finmath.montecarlo.interestrate.modelplugins;

import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretizationInterface;


/**
 * Abstract base class and interface description of a correlation model
 * (as it is used in {@link LIBORCovarianceModelFromVolatilityAndCorrelation}).
 * 
 * Derive from this class and implement the <code>getFactorLoading</code> method.
 * You have to call the constructor of this class to set the time
 * discretizations.
 * 
 * @author Christian Fries
 */
public abstract class LIBORCorrelationModel {
    final TimeDiscretizationInterface	timeDiscretization;
    final TimeDiscretizationInterface	liborPeriodDiscretization;
	
	public LIBORCorrelationModel(TimeDiscretizationInterface timeDiscretization, TimeDiscretizationInterface liborPeriodDiscretization) {
		super();
		this.timeDiscretization = timeDiscretization;
		this.liborPeriodDiscretization = liborPeriodDiscretization;
	}

	/**
	 * Get the parameters of determining this parametric
	 * covariance model. The parameters are usually free parameters
	 * which may be used in calibration.
	 * 
	 * @return Parameter in {@link RandomVariableInterface}-array.
	 */
	public abstract RandomVariableInterface[] getParameterAsRandomVariable();
	
	/**
	 * Get the parameters of determining this parametric
	 * covariance model. The parameters are usually free parameters
	 * which may be used in calibration.
	 * 
	 * @return Parameter in double-array.
	 */
	public double[]	getParameter() {
		// get parameters
		RandomVariableInterface[] parameterAsRandomVariable = getParameterAsRandomVariable();

		// cover case of not calibrateable models
		if(parameterAsRandomVariable == null) return null;

		// get values of deterministic random variables
		double[] parameter = new double[parameterAsRandomVariable.length];
		for(int parameterIndex = 0; parameterIndex < parameterAsRandomVariable.length; parameterIndex++)
			parameter[parameterIndex] = parameterAsRandomVariable[parameterIndex].get(0);

		return parameter;
	}
	
	/**
	 * Get the parameter identifiers of determining this parametric
	 * covariance model, in case parameters are of 
	 * instance {@link RandomVariableDifferentiableInterface}.
	 * 
	 * @return Array of parameter identifiers, null if no internal 
	 * model is calibratable or random variables are not of 
	 * instance {@link RandomVariableDifferentiableInterface}
	 * */
	public long[] getParameterID() {
		RandomVariableInterface[] parameterAsRandomVariable = getParameterAsRandomVariable();
		
		if(parameterAsRandomVariable == null || !(parameterAsRandomVariable[0] instanceof RandomVariableDifferentiableInterface)) return null;
		
		long[] parameterIDs = new long[parameterAsRandomVariable.length];
		for(int parameterIndex = 0; parameterIndex < parameterIDs.length; parameterIndex++)
			parameterIDs[parameterIndex] = ((RandomVariableDifferentiableInterface) parameterAsRandomVariable[parameterIndex]).getID();
		
		return parameterIDs;
	}    public abstract void		setParameter(double[] parameter);

    public abstract	RandomVariableInterface	getFactorLoading(int timeIndex, int factor, int component);
	public abstract	RandomVariableInterface	getCorrelation(int timeIndex, int component1, int component2);
	public abstract int		getNumberOfFactors();

	/**
	 * @return Returns the liborPeriodDiscretization.
	 */
	public TimeDiscretizationInterface getLiborPeriodDiscretization() {
		return liborPeriodDiscretization;
	}

	/**
	 * @return Returns the timeDiscretization.
	 */
	public TimeDiscretizationInterface getTimeDiscretization() {
		return timeDiscretization;
	}

	@Override
    public abstract Object clone();
}
