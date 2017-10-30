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
		this.alpha = 5.0;
		
		this.c1 = 1E-4; /* for linear problems (see p39 in Numerical Optimization) */
		this.c2 = 9E-1; /* 0 < c1 < c2 < 1*/
				
		this.maxStepSize = Math.abs(VectorAlgbra.getAverage(initialParameter));
		this.minStepSize = maxStepSize * 1E-10;
		
		this.hessianInverse = VectorAlgbra.getDiagonalMatrix(1.0, initialParameter.length);
	}
		
	public BroydenFletcherGoldfarbShanno(double[] initialParameters, double targetValue, double errorTolerance, int maxIterations) {
		this(initialParameters, targetValue, errorTolerance, maxIterations, null, null);
	}

	private final double alpha;
	private final double c1; 
	private final double c2; 
	
	private final double maxStepSize;
	private final double minStepSize;


	@Override
	protected double getStepSize(double[] parameter) throws SolverException {

		boolean armijoCondition, wolfeCondition;
			
		double[] parameterCandidate;
		double stepSize = maxStepSize * alpha;
		
		do{
			// try step size candidate
			stepSize /= alpha;
			
			parameterCandidate = VectorAlgbra.add(parameter, VectorAlgbra.scalarProduct(stepSize, currentSearchDirection));
			
			// calculate both sides of the equation for Armijos Condition
			double armijoLeft = getValue(parameterCandidate);
			double armijoRight = currentValue - c1 * stepSize * VectorAlgbra.innerProduct(currentGradient, currentGradient);
			armijoCondition = (armijoLeft <= armijoRight);
			
			// if Armijo's sufficient descent condition not fulfilled continue
			if(!armijoCondition) {
				wolfeCondition = false;
				continue;
			}
			
			double[] gradientCandidate = getDerivative(parameterCandidate); 
			
			// calculate both sides of the equation for the Wolfe Condition
			double wolfeLeft = VectorAlgbra.innerProduct(gradientCandidate, currentSearchDirection);
			double wolfeRight = c2 * VectorAlgbra.innerProduct(currentGradient, currentSearchDirection);			
			wolfeCondition = (wolfeLeft >= wolfeRight);
			
		} while( !(armijoCondition && wolfeCondition) && (stepSize > minStepSize));
				
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
			
			// increase number if iterations
			numberOfIterations++;
			
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
