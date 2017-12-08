package net.finmath.optimizer.gradientdescent;

import java.util.Map;

import net.finmath.functions.VectorAlgbra;
import net.finmath.optimizer.SolverException;

public abstract class GradientDescentArmijosRule extends AbstractGradientDescentScalarOptimization {

	private static final long serialVersionUID = -8770177009110584745L;

	private GradientDescentArmijosRule(
			double[] initialParameter, double targetValue, double errorTolerance,
			int maxNumberOfIterations, 
			long maxRunTime) {
		super(initialParameter, targetValue, errorTolerance, maxNumberOfIterations, maxRunTime, null, null, errorTolerance <= 0.0);

		this.maxStepSize = Math.abs(VectorAlgbra.getAverage(initialParameter)) * 10;
		this.minStepSize = maxStepSize * 1E-15;
		this.lastStepSize = maxStepSize;

		this.alpha = 2.0;
		this.c1 = 1E-4; /* for linear problems (see p39 in Numerical Optimization) */
	}

	public GradientDescentArmijosRule(double[] initialParameters, double targetValue, double errorTolerance, int maxIterations) {
		this(initialParameters, targetValue, errorTolerance, maxIterations, Long.MAX_VALUE);
	}

	public GradientDescentArmijosRule(double[] initialParameters, double targetValue, double errorTolerance, long maxRunTimeInMillis) {
		this(initialParameters, targetValue, errorTolerance, Integer.MAX_VALUE, maxRunTimeInMillis);
	}

	private double alpha;
	private double c1;

	private double maxStepSize;
	private double minStepSize;
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
		while(!isDone() && isStepSizeNeedsAdjustment) {

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

	public GradientDescentArmijosRule cloneWithModifiedParameters(Map<String, Object> properties){

		GradientDescentArmijosRule thisOptimizer = this;

		GradientDescentArmijosRule clone = new GradientDescentArmijosRule(currentParameter, targetValue, errorTolerance,
				maxNumberOfIterations) {

			private static final long serialVersionUID = 1L;

			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
				thisOptimizer.setValues(parameters, values);
			}

			@Override
			public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
				thisOptimizer.setDerivatives(parameters, derivatives);
			}
		};

		// collect general properties
		clone.setProperties(properties, thisOptimizer);

		// collect special properties
		clone.alpha = (double) properties.getOrDefault("alpha",	this.alpha);
		clone.c1 	= (double) properties.getOrDefault("c1",	this.c1);

		clone.maxStepSize 	= (double) properties.getOrDefault("maxStepSize",	this.maxStepSize);
		clone.minStepSize 	= (double) properties.getOrDefault("minStepSize",	this.minStepSize);
		clone.lastStepSize 	= (double) properties.getOrDefault("stepSize",	this.lastStepSize);
		
		return clone;	
	}	
}


