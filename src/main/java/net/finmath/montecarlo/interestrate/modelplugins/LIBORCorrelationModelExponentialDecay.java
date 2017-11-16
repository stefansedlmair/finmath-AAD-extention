/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 20.05.2006
 */
package net.finmath.montecarlo.interestrate.modelplugins;

import net.finmath.functions.LinearAlgebra;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretizationInterface;


/**
 * Simple correlation model given by R, where R is a factor reduced matrix
 * (see {@link net.finmath.functions.LinearAlgebra#factorReduction(double[][], int)}) created from the
 * \( n \) Eigenvectors of \( \tilde{R} \) belonging to the \( n \) largest non-negative Eigenvalues,
 * where \( \tilde{R} = \tilde{\rho}_{i,j} \) and \[ \tilde{\rho}_{i,j} = \exp( -\max(a,0) | T_{i}-T_{j} | ) \]
 * 
 * For a more general model featuring three parameters see {@link LIBORCorrelationModelThreeParameterExponentialDecay}.
 * 
 * @see net.finmath.functions.LinearAlgebra#factorReduction(double[][], int)
 * @see LIBORCorrelationModelThreeParameterExponentialDecay
 * 
 * @author Christian Fries
 */
public class LIBORCorrelationModelExponentialDecay extends LIBORCorrelationModel {

	private final 	AbstractRandomVariableFactory 	randomVariableFactory;
	private 		RandomVariableInterface			a;

	private final	int			numberOfFactors;
	private final	boolean		isCalibrateable;

	private double		matrixParameter;
	private double[][]	correlationMatrix;
	private double[][]	factorMatrix;


	/**
	 * Create a correlation model with an exponentially decaying correlation structure and the given number of factors.
	 * 
	 * @param timeDiscretization Simulation time dicretization. Not used.
	 * @param liborPeriodDiscretization Tenor time discretization, i.e., the \( T_{i} \)'s.
	 * @param numberOfFactors Number \( n \) of factors to be used.
	 * @param a Decay parameter. Should be positive. Negative values will be floored to 0.
	 * @param isCalibrateable If true, the parameter will become a free parameter in a calibration.
	 */
	public LIBORCorrelationModelExponentialDecay(AbstractRandomVariableFactory randomVariableFactory, TimeDiscretizationInterface timeDiscretization, TimeDiscretizationInterface liborPeriodDiscretization, int numberOfFactors, double a, boolean isCalibrateable) {
		super(timeDiscretization, liborPeriodDiscretization);

		this.randomVariableFactory = randomVariableFactory;

		this.numberOfFactors	= numberOfFactors;
		this.isCalibrateable	= isCalibrateable;

		setParameter(new double[]{a}, true);
	}

	public LIBORCorrelationModelExponentialDecay(AbstractRandomVariableFactory randomVariableFactory, TimeDiscretizationInterface timeDiscretization, TimeDiscretizationInterface liborPeriodDiscretization, int numberOfFactors, double a) {
		this(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, numberOfFactors, a, false);
	}

	private void setParameter(double[] parameter, boolean isAllowed){
		if(!isAllowed) return;

		a = randomVariableFactory.createRandomVariable(parameter[0]);

		resetInternalStorage();
		initialize(numberOfFactors, a.doubleValue());
	}


	@Override
	public void setParameter(double[] parameter) {
		setParameter(parameter, isCalibrateable);
	}

	@Override
	public Object clone() {
		return new LIBORCorrelationModelExponentialDecay(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, numberOfFactors, a.doubleValue(), isCalibrateable);
	}

	@Override
	public RandomVariableInterface	getFactorLoading(int timeIndex, int factor, int component) {
		return a.apply( x -> {		
				initialize(numberOfFactors, x);
				return factorMatrix[component][factor];
			});
	}

	@Override
	public RandomVariableInterface	getCorrelation(int timeIndex, int component1, int component2) {
		return a.apply( x -> {
			initialize(numberOfFactors, x);
			return correlationMatrix[component1][component2];});
	}

	@Override
	public int	getNumberOfFactors() {
		return numberOfFactors;
	}

	private void resetInternalStorage(){
		this.matrixParameter = Double.NaN;
		this.correlationMatrix = null;
		this.factorMatrix = null;
	}

	private void initialize(int numberOfFactors, double x) {

		// Negative values of a do not make sense. 
		if(x < 0.0) x = 0.0;


		// if has already been stored 
		if(x == this.matrixParameter) return;
		this.matrixParameter = x;
		/*
		 * Create instantaneous correlation matrix
		 */

		correlationMatrix = new double[liborPeriodDiscretization.getNumberOfTimeSteps()][liborPeriodDiscretization.getNumberOfTimeSteps()];
		for(int row=0; row<correlationMatrix.length; row++) {
			for(int col=0; col<correlationMatrix[row].length; col++) {
				// Exponentially decreasing instantaneous correlation
				correlationMatrix[row][col] = Math.exp(-x * Math.abs(liborPeriodDiscretization.getTime(row)-liborPeriodDiscretization.getTime(col)));
			}
		}

		/*
		 * Perform a factor decomposition (and reduction if numberOfFactors < correlationMatrix.columns())
		 */
		factorMatrix = LinearAlgebra.factorReduction(correlationMatrix, numberOfFactors);

		for(int component1=0; component1<factorMatrix.length; component1++) {
			for(int component2=0; component2<component1; component2++) {
				double correlation = 0.0;
				for(int factor=0; factor<factorMatrix[component1].length; factor++) {
					correlation += factorMatrix[component1][factor] * factorMatrix[component2][factor];
				}
				correlationMatrix[component1][component2] = correlation;
				correlationMatrix[component2][component1] = correlation;
			}
			correlationMatrix[component1][component1] = 1.0;
		}


	}

	@Override
	public RandomVariableInterface[] getParameterAsRandomVariable() {
		return new RandomVariableInterface[]{this.a};
	}


}
