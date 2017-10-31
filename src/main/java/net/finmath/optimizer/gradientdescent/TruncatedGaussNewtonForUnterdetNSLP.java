package net.finmath.optimizer.gradientdescent;

import java.util.concurrent.ExecutorService;

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

	public TruncatedGaussNewtonForUnterdetNSLP(double[] initialParameter, double targetValue, double errorTolerance,
			int maxNumberOfIterations, 
			double[] finiteDifferenceStepSizes, 
			ExecutorService executor) {
		super(initialParameter, targetValue, errorTolerance, maxNumberOfIterations,	finiteDifferenceStepSizes, 
				executor, false);
	}

	public TruncatedGaussNewtonForUnterdetNSLP(double[] initialParameters, double targetValue, double errorTolerance,
			int maxIterations) {
		this(initialParameters, targetValue, errorTolerance, maxIterations, null, null);
	}

	@Override
	protected double getStepSize(double[] parameter) throws SolverException {
		
		// increase number of Iterations
		numberOfIterations++;
		
		double value = getValue(parameter);
		double[] derivative = getDerivative(parameter);
		
		return value / VectorAlgbra.innerProduct(derivative, derivative);
	}
}

