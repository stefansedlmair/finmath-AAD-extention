package net.finmath.optimizer.gradientdescent;

import java.util.concurrent.ExecutorService;

import net.finmath.functions.VectorAlgbra;
import net.finmath.optimizer.SolverException;

public abstract class GradientDescentArmijosRule extends AbstractGradientDescentScalarOptimization {

	private static final long serialVersionUID = -8770177009110584745L;

	public GradientDescentArmijosRule(
			double[] initialParameter, double targetValue, double errorTolerance,
			int maxNumberOfIterations, 
			double[] finiteDifferenceStepSizes, 
			ExecutorService executor, 
			boolean allowWorsening) {
		super(initialParameter, targetValue, errorTolerance, maxNumberOfIterations, finiteDifferenceStepSizes, executor, allowWorsening);
		
		this.maxStepSize = Math.abs(VectorAlgbra.getAverage(initialParameter)) * 10;
		this.minStepSize = maxStepSize * 1E-12;
		this.lastStepSize = maxStepSize;
				
		this.alpha = 2.0;
		this.c1 = 1E-4; /* for linear problems (see p39 in Numerical Optimization) */
	}
	
	public GradientDescentArmijosRule(double[] initialParameters, double targetValue, double errorTolerance, int maxIterations) {
		this(initialParameters, targetValue, errorTolerance, maxIterations, null, null, false);
	}

	private final double alpha;
	private final double c1;
	
	private final double maxStepSize;
	private final double minStepSize;
	private double lastStepSize;

	@Override
	protected double getStepSize(double[] parameter) throws SolverException {
		double leftSide, rightSide ;

		double[] derivative = getDerivative(parameter);
		double derivativeL2Squared = VectorAlgbra.innerProduct(derivative, derivative);
		double 	 value = getValue(parameter);
		
		double stepSize = Math.min(lastStepSize*alpha, maxStepSize);
		
		// Loop will decrease step size before testing.
		stepSize *= alpha;
		boolean isStepSizeNeedsAdjustment = true;
		while(isStepSizeNeedsAdjustment) {
			if(isDone()) break;
			
			// count each line search as iteration
			numberOfIterations++;		
			
			stepSize /= alpha;
			
			double[] newPossibleParameter = VectorAlgbra.subtract(parameter, VectorAlgbra.scalarProduct(stepSize, derivative));
			
			leftSide = getValue(newPossibleParameter);						
			rightSide = value + c1 * stepSize * derivativeL2Squared;

			isStepSizeNeedsAdjustment = (leftSide > rightSide || Double.isNaN(leftSide)) && stepSize > minStepSize;
		};
		
		lastStepSize = stepSize;
		return stepSize;
	}
}


