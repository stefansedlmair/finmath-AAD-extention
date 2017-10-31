package net.finmath.optimizer.gradientdescent;

import java.util.concurrent.ExecutorService;

import net.finmath.functions.VectorAlgbra;
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
	
	private final double fixedStepSize;	
	
	public SimpleGradientDescent(double[] initialParameter, double targetValue, double errorTolerance,
			int maxNumberOfIterations, double[] finiteDifferenceStepSizes, ExecutorService executor,
			boolean allowWorsening, double fixedStepSize) {
		super(initialParameter, targetValue, errorTolerance, maxNumberOfIterations, finiteDifferenceStepSizes, executor,
				allowWorsening);
		
		this.fixedStepSize = fixedStepSize;
	}

	public SimpleGradientDescent(double[] initialParameter, double targetValue, double errorTolerance,
			int maxNumberOfIterations) {
		this(initialParameter, targetValue, errorTolerance, maxNumberOfIterations, null, null, true/*allow optimizer to get worse*/, Math.abs(VectorAlgbra.getAverage(initialParameter) * 1E-1));
	}
	
	
	@Override
	protected double getStepSize(double[] currentParameter) throws SolverException {
		
		// increase number if iterations
		numberOfIterations++;
		
		return fixedStepSize;
	}
	
}


