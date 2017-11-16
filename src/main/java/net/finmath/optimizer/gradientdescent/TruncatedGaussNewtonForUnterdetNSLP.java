package net.finmath.optimizer.gradientdescent;

import java.util.Map;

import net.finmath.functions.VectorAlgbra;
import net.finmath.optimizer.SolverException;

/**
 * Implements the Truncated Gauß-Newton Algorithm
 * stated in the article "<a href="http://www.sciencedirect.com/science/article/pii/S016892741630157X">Approximate Gauss-Newton methods for 
 * solving underdetermined non-linear least squares problems </a>"
 * from J.-F. Bao et al.
 * 
 * It is claimed that this algorithm performs well for a highly underdetermined NLSP.
 * 
 * @author Stefan Sedlmair
 * @version 0.1
 * */
public abstract class TruncatedGaussNewtonForUnterdetNSLP extends  AbstractGradientDescentScalarOptimization{
	
	private static final long serialVersionUID = 7043579549858250182L;

	private TruncatedGaussNewtonForUnterdetNSLP(double[] initialParameter, double targetValue, double errorTolerance,
			int maxNumberOfIterations, long maxRunTime) {
		super(initialParameter, targetValue, errorTolerance, maxNumberOfIterations,	maxRunTime, null, null, errorTolerance <= 0.0);
	}

	public TruncatedGaussNewtonForUnterdetNSLP(double[] initialParameters, double targetValue, double errorTolerance,
			int maxIterations) {
		this(initialParameters, targetValue, errorTolerance, maxIterations, Long.MAX_VALUE);
	}

	@Override
	protected double getStepSize(double[] parameter) throws SolverException {
		
		// increase number of Iterations
		numberOfIterations++;
		
		double value = getValue(parameter);
		double[] derivative = getDerivative(parameter);
			
		return value / VectorAlgbra.innerProduct(derivative, derivative);
	}
	
	public TruncatedGaussNewtonForUnterdetNSLP clone(){
		TruncatedGaussNewtonForUnterdetNSLP thisOptimizer = this;
		
		TruncatedGaussNewtonForUnterdetNSLP clone = new TruncatedGaussNewtonForUnterdetNSLP(currentParameter, targetValue, errorTolerance,
				maxNumberOfIterations) {

			private static final long serialVersionUID = 1L;

			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
				thisOptimizer.setValues(parameters, values);
			}
		};
		return clone;
	}
	
	public TruncatedGaussNewtonForUnterdetNSLP cloneWithModifiedParameters(Map<String, Object> properties){
		
		TruncatedGaussNewtonForUnterdetNSLP thisOptimizer = this;

		TruncatedGaussNewtonForUnterdetNSLP clone = new TruncatedGaussNewtonForUnterdetNSLP(currentParameter, targetValue, errorTolerance,
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

		// collect properties
		clone.setProperties(properties, thisOptimizer);
			
		return clone;	
	}	
}

