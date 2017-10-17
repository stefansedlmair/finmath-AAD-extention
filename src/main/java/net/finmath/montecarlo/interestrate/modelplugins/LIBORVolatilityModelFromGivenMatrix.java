/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 20.05.2006
 */
package net.finmath.montecarlo.interestrate.modelplugins;

import java.util.ArrayList;

import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretizationInterface;

/**
 * Implements a simple volatility model using given piece-wise constant values on
 * a given discretization grid.
 * 
 * @author Christian Fries
 */
public class LIBORVolatilityModelFromGivenMatrix extends LIBORVolatilityModel {

	private final AbstractRandomVariableFactory	randomVariableFactory;
	private final double[][]		volatilityMatrix;
	
	/**
	 * A cache for the parameter associated with this model, it is only used when getParameter is
	 * called repeatedly.
	 */
	private transient RandomVariableInterface[]		parameter = null;	

	// A lazy init cache
	private transient RandomVariableInterface[][] volatility;
	
	/**
	 * Creates a simple volatility model using given piece-wise constant values on
 	 * a given discretization grid.
 	 * 
	 * @param randomVariableFactory The random variable factor used to construct random variables from the parameters. 
	 * @param timeDiscretization Discretization of simulation time.
	 * @param liborPeriodDiscretization Discretization of tenor times.
	 * @param volatility Volatility matrix volatility[timeIndex][componentIndex] where timeIndex the index of the start time in timeDiscretization and componentIndex from liborPeriodDiscretization
	 */
	public LIBORVolatilityModelFromGivenMatrix(
			AbstractRandomVariableFactory randomVariableFactory,
			TimeDiscretizationInterface	timeDiscretization,
			TimeDiscretizationInterface	liborPeriodDiscretization,
			double[][]	volatility) {
		super(timeDiscretization, liborPeriodDiscretization);

		this.randomVariableFactory = randomVariableFactory;
		this.volatilityMatrix = volatility;

		this.volatility = new RandomVariableInterface[timeDiscretization.getNumberOfTimeSteps()][liborPeriodDiscretization.getNumberOfTimeSteps()];
	}

	/**
	 * Creates a simple volatility model using given piece-wise constant values on
 	 * a given discretization grid.
 	 * 
	 * @param timeDiscretization Discretization of simulation time.
	 * @param liborPeriodDiscretization Discretization of tenor times.
	 * @param volatility Volatility matrix volatility[timeIndex][componentIndex] where timeIndex the index of the start time in timeDiscretization and componentIndex from liborPeriodDiscretization
	 */
	public LIBORVolatilityModelFromGivenMatrix(
			TimeDiscretizationInterface	timeDiscretization,
			TimeDiscretizationInterface	liborPeriodDiscretization,
			double[][]	volatility) {
		this(new RandomVariableFactory(), timeDiscretization, liborPeriodDiscretization, volatility);
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModel#getVolatility(int, int)
	 */
	@Override
    public RandomVariableInterface getVolatility(int timeIndex, int component) {
		synchronized (volatility) {
			if(volatility[timeIndex][component] == null) {
				volatility[timeIndex][component] = randomVariableFactory.createRandomVariable(getTimeDiscretization().getTime(timeIndex), volatilityMatrix[timeIndex][component]);
			}
		}

		return volatility[timeIndex][component];
	}

	@Override
	public void setParameter(double[] parameter) {
		this.parameter = null;		// Invalidate cache
		int parameterIndex = 0;
		for(int timeIndex = 0; timeIndex<getTimeDiscretization().getNumberOfTimeSteps(); timeIndex++) {
			for(int liborPeriodIndex = 0; liborPeriodIndex< getLiborPeriodDiscretization().getNumberOfTimeSteps(); liborPeriodIndex++) {
				if(getTimeDiscretization().getTime(timeIndex) < getLiborPeriodDiscretization().getTime(liborPeriodIndex) ) {
					double currentVolatility = parameter[parameterIndex++];
					
					// catch negative values
					if(currentVolatility < 0.0) 
						throw new IllegalArgumentException("Parameter at index " + (parameterIndex - 1) + " indicates negative Volatility(value: "+ currentVolatility +")!");
					
					volatilityMatrix[timeIndex][liborPeriodIndex] = currentVolatility;
				}
			}
		}

		// Invalidate cache
		volatility = new RandomVariableInterface[getTimeDiscretization().getNumberOfTimeSteps()][getLiborPeriodDiscretization().getNumberOfTimeSteps()];

		return;
	}

	@Override
	public Object clone() {
	    // Clone the outer array.
	    double[][] newVolatilityArray = (double[][]) volatilityMatrix.clone();

	    // Clone the contents of the array
	    int rows = newVolatilityArray.length;
	    for(int row=0;row<rows;row++){
	    	newVolatilityArray[row] = (double[]) newVolatilityArray[row].clone();
	    }
			 				
		return new LIBORVolatilityModelFromGivenMatrix(
				randomVariableFactory,
				getTimeDiscretization(),
				getLiborPeriodDiscretization(),
				newVolatilityArray);
	}

	@Override
	public RandomVariableInterface[] getParameterAsRandomVariable() {
		synchronized (this) {
			if(parameter == null) {
				ArrayList<RandomVariableInterface> parameterArray = new ArrayList<>();
				for(int timeIndex = 0; timeIndex<getTimeDiscretization().getNumberOfTimeSteps(); timeIndex++) {
					for(int liborPeriodIndex = 0; liborPeriodIndex< getLiborPeriodDiscretization().getNumberOfTimeSteps(); liborPeriodIndex++) {
						if(getTimeDiscretization().getTime(timeIndex) < getLiborPeriodDiscretization().getTime(liborPeriodIndex) ) {
							parameterArray.add(getVolatility(timeIndex, liborPeriodIndex));
						}
					}
				}
				parameter = new RandomVariableInterface[parameterArray.size()];
				for(int i=0; i<parameter.length; i++) parameter[i] = parameterArray.get(i);
			}
		}

		return parameter;	
	}
}
