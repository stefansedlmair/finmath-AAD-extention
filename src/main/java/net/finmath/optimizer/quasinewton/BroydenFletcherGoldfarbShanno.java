/**
 * 
 */
package net.finmath.optimizer.quasinewton;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import net.finmath.optimizer.OptimizerInterfaceAAD;
import net.finmath.optimizer.SolverException;
import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;

/**
 * Quasi-Newton Algorithm by Broyden Fletcher Goldfarb and Shanno
 * following the discrption of Algorithm 8.1 in "Numerical Optimization" by Nocedal and Wright (see page 198)
 * 
 * 
 * @author Stefan Sedlmair
 *
 */
public abstract class BroydenFletcherGoldfarbShanno implements OptimizerInterfaceAAD,DerivativeFunction, Serializable {

	private static final long serialVersionUID = -2986169210708965199L;

	private RealMatrix hessianInverse;
	private RealVector gradientDelta;
	private RealVector parameterDelta;
	private RealVector functionGradient;
	
	private final double errorTolerance;
	
	private final double targetAccuray;
	private 	  double currentAccuray;
	private 	  double bestAccuray;
	
	private 	  int numberOfIterations;
	private final int maxNumberOfIterations;
	
	private 	  double currentValue;
	private final double targetValue;
	
	private RealVector currentParameters;
	private RealVector bestParameters;
	
	private double	lambda				= 1E-5;
	private double	lambdaDivisor		= 1.1;
	private double	lambdaMultiplicator	= 1.1;
	
	public BroydenFletcherGoldfarbShanno(double[] initialParameters, double targetValue, double targetAccuracy, int maxNumberOfIterations, double errorTolerance) {
		
		int numberOfParameters = initialParameters.length;
		
		this.currentParameters 	= new ArrayRealVector(initialParameters);
		this.bestParameters 	= new ArrayRealVector(initialParameters);
		
		this.errorTolerance = errorTolerance;

		hessianInverse = MatrixUtils.createRealIdentityMatrix(numberOfParameters);
		
		this.targetValue = targetValue;
		
		this.targetAccuray = targetAccuracy;
		this.currentAccuray = Double.POSITIVE_INFINITY;
		this.bestAccuray = Double.POSITIVE_INFINITY;
		
		this.numberOfIterations = 0;
		this.maxNumberOfIterations = maxNumberOfIterations;
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerInterface#getBestFitParameters()
	 */
	@Override
	public double[] getBestFitParameters() {
		return bestParameters.toArray();
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerInterface#getRootMeanSquaredError()
	 */
	@Override
	public double getRootMeanSquaredError() {
		return bestAccuray;
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerInterface#getIterations()
	 */
	@Override
	public int getIterations() {
		return numberOfIterations;
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerInterface#run()
	 */
	@Override
	public void run() throws SolverException {

		boolean lastStepIncreasedAccuracy = true;
		RealVector searchDirection = null;
		// inizialize function gradient
		currentValue = getValue(currentParameters);
		functionGradient = getDerivative(currentParameters);

		while(!isDone()){
			
			// Quasi-Newton Step
			if(lastStepIncreasedAccuracy) 
				searchDirection = hessianInverse.operate(functionGradient).mapMultiply(-1.0);
			
			// step size (possibly by wolfe condition but for simplicity like in LM)			
			double stepSize = getStepSize(searchDirection);
			
			// update parameters
			parameterDelta = searchDirection.mapMultiply(stepSize).copy();
			currentParameters = currentParameters.add(parameterDelta).copy();
			
			// evaluate new parameters
			currentValue = getValue(currentParameters);
			currentAccuray = Math.abs(currentValue - targetValue);
			
			System.out.println("step# " + numberOfIterations + "\t accuracy: " + currentAccuray );
			
			if(currentAccuray < bestAccuray){
				// update best performance
				bestParameters = currentParameters.copy();
				bestAccuray = currentAccuray;
				
				// reduce step size
				lambda /= lambdaDivisor;
				
				// last step was better than before
				lastStepIncreasedAccuracy = true;
			} else {
				// increase stepsize
				lambda *= lambdaMultiplicator;
				
				// last step was worse than before
				lastStepIncreasedAccuracy = false;
			}
			
			if(lastStepIncreasedAccuracy){
				// update hessian
				RealVector oldFunctionGradient = functionGradient.copy();
				functionGradient = getDerivative(currentParameters);
				gradientDelta = functionGradient.subtract(oldFunctionGradient).copy();
				
				double hessianInverseAddend1Divisor = 1.0/gradientDelta.dotProduct(hessianInverse.operate(gradientDelta));
				double hessianInverseAddend2Divisor = 1.0/gradientDelta.dotProduct(parameterDelta);
				
				RealMatrix hessianInverseAddend1 = hessianInverse.operate(gradientDelta).outerProduct(gradientDelta).multiply(hessianInverse);
				RealMatrix hessianInverseAddend2 = parameterDelta.outerProduct(parameterDelta);
				
				hessianInverse = hessianInverse.subtract(hessianInverseAddend1.scalarMultiply(hessianInverseAddend1Divisor)).subtract(hessianInverseAddend2.scalarMultiply(hessianInverseAddend2Divisor));
			}
			
			// increase number of iterations
			numberOfIterations++;
		}
		
	}
	
	public enum StepSizeType{
		WolfeCondition, Constant
	}
	
	private StepSizeType stepSizeType = StepSizeType.Constant;
	
	private double getStepSize(RealVector searchDirection) throws SolverException{
		double stepSize = Double.NaN;
		switch (stepSizeType) {
		case WolfeCondition:
			double l = 0;
			double newValue;
			RealVector parameterCandidate, newGradient;
			boolean condition1 = false; 
			boolean condition2 = false;
			// constants from book (see Wolfe Condition)
			double c1 = 1E-4;
			double c2 = 0.9;
			
			do{
				//stepSize progression like for Armijos Rule
				stepSize = Math.pow(lambda, l++);
				
				// new parameter candidate
				parameterCandidate = currentParameters.add(searchDirection.mapMultiply(stepSize));
				
				// get value and derivative for new parameter candidates
				newValue = getValue(parameterCandidate);
				newGradient = getDerivative(parameterCandidate);	
				
				if(Double.isNaN(newValue) || newGradient.isNaN()) continue;
				
				condition1 = (newValue <= (currentValue - c1 * stepSize * functionGradient.dotProduct(functionGradient)));
				condition2 = ((-newGradient.dotProduct(functionGradient)) >= (-c2 * functionGradient.dotProduct(functionGradient)));
			} while(!(condition1 && condition2) && stepSize > 1E-8);
			break;

		case Constant:
			stepSize = lambda;
			break;
		default:
			stepSize = lambda;
		}
			
		return stepSize;
	}
	
	public boolean isDone(){
		return 
				bestAccuray < targetAccuray
				||
				numberOfIterations > maxNumberOfIterations
				||
				functionGradient.getNorm() < errorTolerance;
	}

	@Override
	public abstract void setValues(double[] parameters, double[] values) throws SolverException;

	@Override
	public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
		// Calculate new derivatives. Note that this method is called only with
		// parameters = parameterCurrent, so we may use valueCurrent.

		Vector<Future<double[]>> valueFutures = new Vector<Future<double[]>>(currentParameters.getDimension());
		for (int parameterIndex = 0; parameterIndex < currentParameters.getDimension(); parameterIndex++) {
			final double[] parametersNew	= parameters.clone();
			final double[] derivative		= derivatives[parameterIndex];

			final int workerParameterIndex = parameterIndex;
			Callable<double[]> worker = new  Callable<double[]>() {
				public double[] call() throws SolverException {
					
					double parameterFiniteDifference = (Math.abs(parametersNew[workerParameterIndex]) + 1) * 1E-8;

					// Shift parameter value
					parametersNew[workerParameterIndex] += parameterFiniteDifference;

					// Calculate derivative as (valueUpShift - valueCurrent) / parameterFiniteDifference
					try {
						setValues(parametersNew, derivative);
					} catch (Exception e) {
						// We signal an exception to calculate the derivative as NaN
						Arrays.fill(derivative, Double.NaN);
					}

					derivative[0] -= currentValue;
					derivative[0] /= parameterFiniteDifference;
					if(Double.isNaN(derivative[0])) derivative[0] = 0.0;

					return derivative;
				}
			};
			
			FutureTask<double[]> valueFutureTask = new FutureTask<double[]>(worker);
			valueFutureTask.run();
			valueFutures.add(parameterIndex, valueFutureTask);
		}

		for (int parameterIndex = 0; parameterIndex < currentParameters.getDimension(); parameterIndex++) {
			try {
				derivatives[parameterIndex] = valueFutures.get(parameterIndex).get();
			}
			catch (InterruptedException e) {
				throw new SolverException(e);
			} catch (ExecutionException e) {
				throw new SolverException(e);
			}
		}
			


		for (int parameterIndex = 0; parameterIndex < currentParameters.getDimension(); parameterIndex++) {
			try {
				derivatives[parameterIndex] = valueFutures.get(parameterIndex).get();
			}
			catch (InterruptedException e) {
				throw new SolverException(e);
			} catch (ExecutionException e) {
				throw new SolverException(e);
			}
		}
	}


	
	private double getValue(RealVector parameter) throws SolverException{
		double[] values = new double[1];
		setValues(parameter.toArray(), values);
		return values[0];
	}
	
	private RealVector getDerivative(RealVector parameter) throws SolverException{
		double[][] derivative = new double[currentParameters.getDimension()][1 /*only one dimension*/];
		setDerivatives(parameter.toArray(), derivative);
		return (new Array2DRowRealMatrix(derivative, false /*copy array*/)).getColumnVector(0);
	}
}
