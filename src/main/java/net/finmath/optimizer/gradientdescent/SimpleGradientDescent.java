package net.finmath.optimizer.gradientdescent;

import java.util.Map;

import net.finmath.optimizer.SolverException;

/**
 * Simple Gradient Descent Optimizer with constant step size, i.e. the update rule is:
 * 
 * <center> x<sub>k+1</sub> = x<sub>k</sub> - &lambda * &nabla f(x<sub>k</sub>) </center>
 * for k=0,1,2,... until accuracy does not become better anymore.
 * 
 * @author Stefan Sedlmair
 * @version 1.0
 * 
 * @see AbstractGradientDescentScalarOptimization
 * */
public abstract class SimpleGradientDescent extends AbstractGradientDescentScalarOptimization {

	private static final long serialVersionUID = -84822697392025037L;
	
	private double fixedStepSize;	
	
	private SimpleGradientDescent(double[] initialParameter, double targetValue, double errorTolerance,
			int maxNumberOfIterations, long maxRunTime ,double fixedStepSize) {
		super(initialParameter, targetValue, errorTolerance, maxNumberOfIterations, maxRunTime, null, null, errorTolerance <= 0.0);
		
		this.fixedStepSize = fixedStepSize;
	}

	public SimpleGradientDescent(double[] initialParameter, double targetValue, double errorTolerance,
			int maxNumberOfIterations) {
		this(initialParameter, targetValue, errorTolerance, maxNumberOfIterations, Long.MAX_VALUE, 1E-3);
	}
	
	public SimpleGradientDescent(double[] initialParameter, double targetValue, double errorTolerance,
			long maxRunTimeInMillis) {
		this(initialParameter, targetValue, errorTolerance, Integer.MAX_VALUE, maxRunTimeInMillis, 1E-3);
	}
	
	@Override
	protected double getStepSize(double[] parameter) throws SolverException {
	
		double stepSize = fixedStepSize;

		numberOfIterations++;
		
		return stepSize;
	}
	
	public SimpleGradientDescent cloneWithModifiedParameters(Map<String, Object> properties){

		SimpleGradientDescent thisOptimizer = this;

		SimpleGradientDescent clone = new SimpleGradientDescent(currentParameter, targetValue, errorTolerance,
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
		clone.fixedStepSize = (double) properties.getOrDefault("stepSize",	this.fixedStepSize);
		
		return clone;	
	}	
	
}


