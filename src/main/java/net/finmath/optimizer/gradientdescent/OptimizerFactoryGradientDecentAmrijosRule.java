package net.finmath.optimizer.gradientdescent;

import net.finmath.optimizer.OptimizerFactoryInterface;
import net.finmath.optimizer.OptimizerInterface;
import net.finmath.optimizer.OptimizerInterfaceAAD;
import net.finmath.optimizer.SolverException;
import net.finmath.optimizer.OptimizerInterface.ObjectiveFunction;
import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;

public class OptimizerFactoryGradientDecentAmrijosRule implements OptimizerFactoryInterface {

	private final int		maxIterations;
	private final double	targetAccuracy;
	private final double	minAdvancePerStep;

	
	public OptimizerFactoryGradientDecentAmrijosRule(int maxIterations,double minAdvancePerStep ,double targetAccuracy) {
		super();
		this.maxIterations = maxIterations;
		this.targetAccuracy = targetAccuracy;
		this.minAdvancePerStep = minAdvancePerStep;
	}
	
	@Override
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters,
			double[] targetValues) {
		return new GradientDescentAmrijosRule(initialParameters, targetValues[0], targetAccuracy, minAdvancePerStep, maxIterations) {

			private static final long serialVersionUID = 4815986388931167776L;

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
		return new GradientDescentAmrijosRule(initialParameters, targetValues[0], targetAccuracy, minAdvancePerStep, maxIterations) {

			private static final long serialVersionUID = 4815986388931167776L;

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
