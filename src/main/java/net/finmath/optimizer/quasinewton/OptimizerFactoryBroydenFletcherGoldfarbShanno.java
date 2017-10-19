/**
 * 
 */
package net.finmath.optimizer.quasinewton;

import net.finmath.optimizer.OptimizerFactoryInterface;
import net.finmath.optimizer.OptimizerInterface;
import net.finmath.optimizer.OptimizerInterface.ObjectiveFunction;
import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;
import net.finmath.optimizer.SolverException;

/**
 * 
 * 
 * @author Stefan Sedlmair
 * @version 0.1
 */
public class OptimizerFactoryBroydenFletcherGoldfarbShanno implements OptimizerFactoryInterface {


	private final int		maxIterations;
	private final double	targetAccuracy;
	private final double	errorTolerance;

	
	public OptimizerFactoryBroydenFletcherGoldfarbShanno(int maxIterations,double errorTolerance ,double targetAccuracy) {
		super();
		this.maxIterations = maxIterations;
		this.targetAccuracy = targetAccuracy;
		this.errorTolerance = errorTolerance;
	}
	
	@Override
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters,
			double[] targetValues) {
		return new BroydenFletcherGoldfarbShanno(initialParameters, targetValues[0], targetAccuracy, maxIterations, errorTolerance) {

			private static final long serialVersionUID = 7815089222413087835L;

			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
				objectiveFunction.setValues(parameters, values);				
			}
		};
	}

	@Override
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters,
			double[] lowerBound, double[] upperBound, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, targetValues);
	}

	@Override
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters,
			double[] lowerBound, double[] upperBound, double[] parameterStep, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, targetValues);

	}

	@Override
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters,
			double[] lowerBound, double[] upperBound, double[] parameterSteps, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, targetValues);
	}

	@Override
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters,
			double[] targetValues) {
		return new BroydenFletcherGoldfarbShanno(initialParameters, targetValues[0], targetAccuracy, maxIterations, errorTolerance) {

			private static final long serialVersionUID = -6683749522552374617L;

			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
				objectiveFunction.setValues(parameters, values);				
			}
			
			@Override
			public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
				objectiveFunction.setDerivatives(parameters, derivatives);
			}
		};
	}

	@Override
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters,
			double[] lowerBound, double[] upperBound, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, targetValues);
	}
}
