/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christianfries.com.
 *
 * Created on 29.05.2015
 */

package net.finmath.optimizer;

import java.util.Arrays;

import net.finmath.optimizer.OptimizerInterface.ObjectiveFunction;
import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;

/**
 * @author Christian Fries
 * @author Stefan Sedlmair
 */
public class OptimizerFactoryLevenbergMarquardt implements OptimizerFactoryInterface {
	
	private final int		maxIterations;
	private final double	errorTolerance;
	private final int		maxThreads;

	public OptimizerFactoryLevenbergMarquardt(int maxIterations, double errorTolerance, int maxThreads) {
		super();
		this.maxIterations = maxIterations;
		this.errorTolerance = errorTolerance;
		this.maxThreads = maxThreads;
	}

	public OptimizerFactoryLevenbergMarquardt(int maxIterations, int maxThreads) {
		this(maxIterations, 0.0, maxThreads);
	}

	@Override
	public OptimizerInterface getOptimizer(final ObjectiveFunction objectiveFunction, final double[] initialParameters, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, null, null, null, targetValues);
	}

	@Override
	public OptimizerInterface getOptimizer(final ObjectiveFunction objectiveFunction, final double[] initialParameters, final double[] lowerBound,final double[]  upperBound, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, lowerBound, upperBound, null, targetValues);
	}

	@Override
	public OptimizerInterface getOptimizer(final ObjectiveFunction objectiveFunction, double[] initialParameters, double[] lowerBound,double[]  upperBound, double[] parameterSteps, double[] targetValues) {
		return (new LevenbergMarquardt(
				initialParameters,
				targetValues,
				maxIterations,
				maxThreads)
		{	
			private static final long serialVersionUID = -1628631567190057495L;

			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
				objectiveFunction.setValues(parameters, values);
			}
			
//			@Override
//			public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
//				super.setDerivatives(parameters, derivatives);
//				
//				System.out.println("FD:");
//				for(double[] row : derivatives) System.out.println(Arrays.toString(row));
//				System.out.println();
//			}
		})
				.setErrorTolerance(errorTolerance)
				.setParameterSteps(parameterSteps);
	}
	
	@Override
	public OptimizerInterface getOptimizer(final DerivativeFunction objectiveFunction, double[] initialParameters, double[] lowerBound,double[]  upperBound, double[] parameterSteps, double[] targetValues) {
		return (new LevenbergMarquardt(
				initialParameters,
				targetValues,
				maxIterations,
				maxThreads)
		{	
			private static final long serialVersionUID = -1628631567190057495L;
			
			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
				objectiveFunction.setValues(parameters, values);
			}
			
			@Override
			public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
				objectiveFunction.setDerivatives(parameters, derivatives);
				
//				System.out.println("AAD:");
//				for(double[] row : derivatives) System.out.println(Arrays.toString(row));
//				System.out.println();
			}
		})
				.setErrorTolerance(errorTolerance)
				.setParameterSteps(parameterSteps);
	}
	
	@Override
	public OptimizerInterface getOptimizer(final DerivativeFunction objectiveFunction, final double[] initialParameters, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, null, null, null, targetValues);
	}

	@Override
	public OptimizerInterface getOptimizer(final DerivativeFunction objectiveFunction, final double[] initialParameters, final double[] lowerBound,final double[]  upperBound, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, lowerBound, upperBound, null, targetValues);
	}
}
