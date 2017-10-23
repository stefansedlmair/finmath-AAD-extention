/**
 * 
 */
package net.finmath.optimizer.quasinewton;

import java.util.concurrent.ExecutorService;

import net.finmath.functions.VectorAlgbra;
import net.finmath.optimizer.SolverException;
import net.finmath.optimizer.gradientdescent.AbstractGradientDescentScalarOptimization;

/**
 * Quasi-Newton Algorithm by Broyden Fletcher Goldfarb and Shanno
 * following the discrption of Algorithm 8.1 in "Numerical Optimization" by Nocedal and Wright (see page 198)
 * 
 * @author Stefan Sedlmair
 */
public abstract class BroydenFletcherGoldfarbShanno extends AbstractGradientDescentScalarOptimization {

	private static final long serialVersionUID = 8690381643877867542L;

	private 	  double[][] hessianInverse;
	private 	  double[]   currentSearchDirection;
	private 	  double[]   currentGradient;
	private final double 	 convergenceTolerance;
	
	public BroydenFletcherGoldfarbShanno(double[] initialParameter, double targetValue, double errorTolerance,	int maxNumberOfIterations, double[] finiteDifferenceStepSizes, ExecutorService executor) {
		super(initialParameter, targetValue, errorTolerance, maxNumberOfIterations, finiteDifferenceStepSizes, executor, false);
		
		this.convergenceTolerance = 0.0;
		this.hessianInverse = VectorAlgbra.getDiagonalMatrix(1.0, initialParameter.length);
	}
		
	public BroydenFletcherGoldfarbShanno(double[] initialParameters, double targetValue, int maxIterations, double errorTolerance) {
		this(initialParameters, targetValue, errorTolerance, maxIterations, null, null);
	}

	private final double alpha				= 1E-1;	/* no good value suggested */
	private final double c1					= 1E-4; /* for linear problems (see p39 in Numerical Optimization) */
	private final double c2					= 9E-1; /* 0 < c1 < c2 < 1*/
	
	private final int maxL 					= 10;


	@Override
	protected double getStepSize(double[] parameter) throws SolverException {

		double armijoLeft, armijoRight;
		double wolfeLeft  = Double.NaN;
		double wolfeRight = Double.NaN;
		
		double l = 0; 
			
		double[] parameterCandidate;
		double stepSize = Double.NaN;
		
		do{
			// try step size candidate
			stepSize = Math.pow(alpha, l++);
			
			parameterCandidate = VectorAlgbra.add(parameter, VectorAlgbra.scalarProduct(stepSize, currentSearchDirection));
			
			armijoLeft = getValue(parameterCandidate);
			armijoRight = currentValue - c1 * stepSize * VectorAlgbra.innerProduct(currentGradient, currentGradient);
			
			// don't check Wolfe Condition if Armijo's condition is not fulfilled anyway
			// NOTE: computational cost of Wolfe condition ~50x higher than Armijo's condition
			if(armijoLeft > armijoRight) continue;
			
			double[] candidateGradient = getDerivative(parameterCandidate); 
			
			wolfeLeft = VectorAlgbra.innerProduct(candidateGradient, currentSearchDirection);
			wolfeRight = c2 * VectorAlgbra.innerProduct(currentGradient, currentSearchDirection);
			
		} while(armijoLeft > armijoRight && wolfeLeft < wolfeRight && l <= maxL);
		
		System.out.println("l = " + (l-1));
		
		return stepSize;
	}
	
	@Override
	public void run() throws SolverException {
		
		// check accuracy of initial parameter
		currentValue = getValue(currentParameter);
		currentAccuracy = Math.abs(currentValue - targetValue);
		bestAccuracy = currentAccuracy;
		
		// additionally check gradient for initial parameters
		currentGradient = getDerivative(currentParameter);
		
		while(!isDone() && (VectorAlgbra.normL2(currentGradient) > convergenceTolerance)){
			
			// calculate new search direction 
			currentSearchDirection = VectorAlgbra.scalarProduct(-1.0, VectorAlgbra.dotProduct(hessianInverse, currentGradient));
			
			// get new step size
			double stepSize = getStepSize(currentParameter);
			
			// save last values for later
			double[] lastParameterSet = currentParameter.clone();
			double[] lastGradient = currentGradient.clone();
			
			// update parameters
			currentParameter = VectorAlgbra.add(currentParameter, VectorAlgbra.scalarProduct(stepSize, currentSearchDirection));
			
			// update value and accuracy
			currentValue = getValue(currentParameter);
			currentAccuracy = Math.abs(currentValue - targetValue);
			
			if(currentAccuracy < bestAccuracy){
				
				// update best parameter and accuracy
				bestParameter = currentParameter.clone();
				bestAccuracy = currentAccuracy;
				
			} else break;
			// if last iterations did not increase accuracy this part will not be executed!
			
			// update gradient for updated parameter set
			currentGradient = getDerivative(currentParameter);
			
			// calculate changes in parameter and gradients
			double[] parameterDelta = VectorAlgbra.subtract(currentParameter, lastParameterSet);
			double[] gradientDelta = VectorAlgbra.subtract(lastGradient, currentGradient);
			
			// calculate weight matrix for hessian update
			double rho = VectorAlgbra.innerProduct(gradientDelta, parameterDelta);
			double[][] weightMatrix = VectorAlgbra.outerProduct(parameterDelta, gradientDelta);
			double[][] identityMatrix = VectorAlgbra.getDiagonalMatrix(1.0, currentParameter.length);
			
			weightMatrix = VectorAlgbra.subtract(identityMatrix, VectorAlgbra.scalarDivision(weightMatrix, rho));
			
			// calculate addend matrix for hessian inverse
			double[][] addendMatrix = VectorAlgbra.scalarDivision(VectorAlgbra.outerProduct(parameterDelta, parameterDelta), rho);
			
			// update hessian matrix
			hessianInverse = VectorAlgbra.dotProduct(VectorAlgbra.dotProduct(weightMatrix, hessianInverse), weightMatrix);
			hessianInverse = VectorAlgbra.add(hessianInverse, addendMatrix);
		}
		
	}
}
