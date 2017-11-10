/**
 * 
 */
package net.finmath.optimizer.quasinewton;

import java.util.Map;

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

	private double[][] hessianInverse;
	private double[]   currentSearchDirection;
	private double[]   currentGradient;
	private double 	 convergenceTolerance;

	private BroydenFletcherGoldfarbShanno(double[] initialParameter, double targetValue, double errorTolerance,	int maxNumberOfIterations, long maxRunTime) {
		super(initialParameter, targetValue, errorTolerance, maxNumberOfIterations, maxRunTime, null, null, errorTolerance <= 0.0);

		this.convergenceTolerance = 0.0;

		lambdaDivisor		= 2.0;
		lambdaMultiplicator	= 3.0;

		this.c1 = 1E-5; /* for linear problems (see p39 in Numerical Optimization) */
		this.c2 = 1E-4;//9E-1; /* 0 < c1 < c2 < 1*/

		this.maxStepSize = 1.0;//Math.abs(VectorAlgbra.getAverage(initialParameter));
		this.minStepSize = maxStepSize * 1E-10;

		this.hessianInverse = VectorAlgbra.getDiagonalMatrix(1.0, initialParameter.length);
	}

	public BroydenFletcherGoldfarbShanno(double[] initialParameters, double targetValue, double errorTolerance, int maxIterations) {
		this(initialParameters, targetValue, errorTolerance, maxIterations, Long.MAX_VALUE);
	}

	public BroydenFletcherGoldfarbShanno(double[] initialParameters, double targetValue, double errorTolerance, long maxRunTimeInMillis) {
		this(initialParameters, targetValue, errorTolerance, Integer.MAX_VALUE, maxRunTimeInMillis);
	}

	private double	lambdaDivisor;
	private double	lambdaMultiplicator;

	private double c1; 
	private double c2; 

	private double maxStepSize;
	private double minStepSize;

	private double lastStepSize = Double.NaN;

	@Override
	protected double getStepSize(double[] parameter) throws SolverException {

		boolean armijoCondition, wolfeCondition;

		double[] parameterCandidate;
		double stepSize = (Double.isNaN(lastStepSize) ? maxStepSize : lastStepSize) * lambdaDivisor;

		do{		
			// increase number if iterations; each line search is considered one iteration
			numberOfIterations++;

			// try step size candidate
			stepSize /= lambdaDivisor;

			//			System.out.println("try stepping " + stepSize);

			parameterCandidate = VectorAlgbra.add(parameter, VectorAlgbra.scalarProduct(stepSize, currentSearchDirection));

			// calculate both sides of the equation for Armijos Condition
			double armijoLeft = getValue(parameterCandidate);
			double armijoRight = currentValue - c1 * stepSize * VectorAlgbra.innerProduct(currentGradient, currentGradient);
			armijoCondition = (armijoLeft <= armijoRight);

			// if Armijo's sufficient descent condition not fulfilled continue
			if(!armijoCondition) {
				wolfeCondition = false;
			} else {
				double[] gradientCandidate = getDerivative(parameterCandidate); 

				// calculate both sides of the equation for the Wolfe Condition
				double wolfeLeft = VectorAlgbra.innerProduct(gradientCandidate, currentSearchDirection);
				double wolfeRight = c2 * VectorAlgbra.innerProduct(currentGradient, currentSearchDirection);			
				wolfeCondition = (wolfeLeft >= wolfeRight);
			}			
			System.out.println(numberOfIterations + ";" + (System.currentTimeMillis() - startTimeInMilliSeconds) +";"+ currentAccuracy);
		} while(!isDone() && !(armijoCondition && wolfeCondition) && (stepSize > minStepSize));

		// give algorithm the chance to accelerate!
		lastStepSize = stepSize * lambdaMultiplicator; 		

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

		startTimeInMilliSeconds = System.currentTimeMillis();

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

			System.out.println(numberOfIterations + ";" + (System.currentTimeMillis() - startTimeInMilliSeconds) +";"+ currentAccuracy);

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

	public BroydenFletcherGoldfarbShanno cloneWithModifiedParameters(Map<String, Object> properties){

		BroydenFletcherGoldfarbShanno thisOptimizer = this;

		BroydenFletcherGoldfarbShanno clone = new BroydenFletcherGoldfarbShanno(currentParameter, targetValue, errorTolerance,
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
		clone.lambdaDivisor 		= (double) 	properties.getOrDefault("lambdaDivisor",this.lambdaDivisor);
		clone.lambdaMultiplicator 	= (double) 	properties.getOrDefault("lambdaMultiplicator",this.lambdaMultiplicator);

		clone.c1 = (double) properties.getOrDefault("c1",this.c1);
		clone.c2 = (double) properties.getOrDefault("c2",this.c2);

		clone.maxStepSize = (double) properties.getOrDefault("maxStepSize",this.maxStepSize);
		clone.minStepSize = (double) properties.getOrDefault("minStepSize",this.minStepSize);

		clone.convergenceTolerance = (double) properties.getOrDefault("convergenceTolerance",this.convergenceTolerance);

		return clone;	
	}	

}
