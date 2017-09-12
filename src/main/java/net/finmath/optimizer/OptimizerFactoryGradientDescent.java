/**
 * 
 */
package net.finmath.optimizer;

import net.finmath.optimizer.OptimizerInterface.ObjectiveFunction;
import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;

/**
 * @author Alois Gugelhupfer
 *
 */
public class OptimizerFactoryGradientDescent implements OptimizerFactoryInterface {

	private final int		maxIterations;
	private final double	errorTolerance;
	private final int		maxThreads;
	
	public OptimizerFactoryGradientDescent(int maxIterations, double errorTolerance, int maxThreads) {
		super();
		this.maxIterations = maxIterations;
		this.errorTolerance = errorTolerance;
		this.maxThreads = maxThreads;
	}
	
	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterface.ObjectiveFunction, double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, null, null, targetValues);
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterface.ObjectiveFunction, double[], double[], double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters,	double[] lowerBound, double[] upperBound, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, null, null, null, targetValues);
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterface.ObjectiveFunction, double[], double[], double[], double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters,	double[] lowerBound, double[] upperBound, double[] finiteDifferenceStepSize, double[] targetValues) {
		return (new GradientDescent(initialParameters, targetValues, errorTolerance, maxIterations) {

			private static final long serialVersionUID = -597832728090163557L;

			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
				objectiveFunction.setValues(parameters, values);			
			}
		});
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction, double[], double[], double[], double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters, double[] lowerBound, double[] upperBound, double[] finiteDifferenceStepSizes, double[] targetValues) {
		return (new GradientDescent(initialParameters, targetValues, errorTolerance, maxIterations) {

			private static final long serialVersionUID = 8937061844543179013L;

			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
				objectiveFunction.setValues(parameters, values);			
			}
			
			@Override
			public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
				objectiveFunction.setDerivatives(parameters, derivatives);
			}
		});
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction, double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, null,  null,  null, targetValues);
	}
		

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction, double[], double[], double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters, double[] lowerBound, double[] upperBound, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, lowerBound,  upperBound,  null, targetValues);
	}

}
