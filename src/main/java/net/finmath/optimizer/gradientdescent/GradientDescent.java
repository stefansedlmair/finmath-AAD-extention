package net.finmath.optimizer.gradientdescent;

import java.util.concurrent.ExecutorService;

import net.finmath.optimizer.SolverException;

/**
 * Simple Gradient Descent Optimizer
 * 
 * with varying step size:
 * <lu>
 * <li> if last step better than best result until now:
 * lambda<sub>k+1</sub> = lambda<sub>k</sub> / lambda<sub>Divisor</sub>,
 * </li>
 * <li> if last step worse than best result until now:
 * lambda<sub>k+1</sub>  = lambda<sub>k</sub>  * lambda<sub>Multiplicator</sub>,
 * </li>
 * </lu>
 * where lambda<sub>Divisor</sub> and lambda<sub>Divisor</sub> are greater than 1.0.
 * */
public abstract class GradientDescent extends AbstractGradientDescentScalarOptimization {

	public GradientDescent(double[] initialParameter, double targetValue, double errorTolerance,
			int maxNumberOfIterations, double[] finiteDifferenceStepSizes, ExecutorService executor,
			boolean allowWorsening) {
		super(initialParameter, targetValue, errorTolerance, maxNumberOfIterations, finiteDifferenceStepSizes, executor,
				allowWorsening);
	}

	public GradientDescent(double[] initialParameter, double targetValue, double errorTolerance,
			int maxNumberOfIterations) {
		this(initialParameter, targetValue, errorTolerance, maxNumberOfIterations, null, null, true);
		}
	
	private static final long serialVersionUID = -84822697392025037L;

	private double	lambda				= 1E-2;
	
	private double	lambdaDivisor		= 1.3;
	private double	lambdaMultiplicator	= 1.2;

	
	private double lastAccuracy = Double.POSITIVE_INFINITY;
	
	@Override
	protected double getStepSize(double[] currentParameter) throws SolverException {

		double stepSize = lambda;
		
		lambda = (currentAccuracy < lastAccuracy) ? lambda * lambdaMultiplicator : lambda / lambdaDivisor; 
	
		lastAccuracy = currentAccuracy;

		return stepSize;
	}
	
}


