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
	}
	
	public GradientDescentArmijosRule(double[] initialParameters, double targetValue, double errorTolerance, int maxIterations) {
		super(initialParameters, targetValue, errorTolerance, maxIterations, null, null, false);
	}

	private final double alpha				= 1E-1;	/* no good value suggested */
	private final double c1					= 1E-4; /* for linear problems (see p39 in Numerical Optimization) */
	
	private final int maxL 					= 10;

	@Override
	protected double getStepSize(double[] parameter) throws SolverException {
		double leftSide, rightSide ;
		double l = 3; // start with l=4 to jump not too fast

		double[] derivative = getDerivative(parameter);
		double 	 value = getValue(parameter);
		
		double[] newPossibleParameter;
		double betaL = Double.NaN;
		
		do{
			betaL = Math.pow(alpha, l++);
			
			newPossibleParameter = VectorAlgbra.subtract(parameter, VectorAlgbra.scalarProduct(betaL, derivative));
			
			leftSide = getValue(newPossibleParameter);
						
			rightSide = value - c1 * betaL * VectorAlgbra.innerProduct(derivative, derivative);
			
		} while((leftSide > rightSide || Double.isNaN(leftSide)) && l <= maxL);
		
		System.out.println("l = " + l);
		
		return betaL;
	}
}


