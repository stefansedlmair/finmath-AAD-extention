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
		
		this.maxStepSize = Math.abs(VectorAlgbra.getAverage(initialParameter));
		this.minStepSize = maxStepSize * 1E-10;
				
		this.alpha = 5.0;
		this.c1 = 1E-4; /* for linear problems (see p39 in Numerical Optimization) */
	}
	
	public GradientDescentArmijosRule(double[] initialParameters, double targetValue, double errorTolerance, int maxIterations) {
		this(initialParameters, targetValue, errorTolerance, maxIterations, null, null, false);
	}

	private final double alpha;
	private final double c1;
	
	private final double maxStepSize;
	private final double minStepSize;

	@Override
	protected double getStepSize(double[] parameter) throws SolverException {
		double leftSide, rightSide ;

		double[] derivative = getDerivative(parameter);
		double 	 value = getValue(parameter);
		
		double[] newPossibleParameter;
		double stepSize = maxStepSize * alpha;
		
		do{
			if(isDone()) break;
			
			// count each line search as iteration
			numberOfIterations++;		
			
			stepSize /= alpha;
			
			newPossibleParameter = VectorAlgbra.subtract(parameter, VectorAlgbra.scalarProduct(stepSize, derivative));
			
			leftSide = getValue(newPossibleParameter);
						
			rightSide = value - c1 * stepSize * VectorAlgbra.innerProduct(derivative, derivative);
			
		} while((leftSide > rightSide || Double.isNaN(leftSide)) && stepSize > minStepSize);
		
		return stepSize;
	}
}


