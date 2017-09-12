package net.finmath.optimizer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;

import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;

public abstract class GradientDescent implements Serializable, Cloneable, OptimizerInterface, DerivativeFunction {

	private static final long serialVersionUID = -84822697392025037L;

	private double targetAccuracy;
	private double currentAccuracy		= Double.POSITIVE_INFINITY;
	private double bestAccuracy 		= Double.POSITIVE_INFINITY;
	private int numberOfIterations		= 0;
	private int maxNumberOfIterations;


	private double[] bestParameters;
	private double[] currentParameters;

	private double[] finiteDifferenceStepSizes;

	private double[] targetValues;
	private double[] currentValues;
	private double[] weights;

	private double	lambda				= 0.5;
	private double  minLambda			= 1E-6;
	private double  maxLambda			= 1E+6;
	
	private double	lambdaDivisor		= 1.1;
	private double	lambdaMultiplicator	= 2.0;

	private double[] derivativeStorage;

	private ExecutorService executor;

	public GradientDescent(double[] initialParameters, double[] targetValues, double[] weights, double[] finiteDifferenceStepSizes,
			double targetAccuracy, int maxNumberOfIterations) {
		this.bestParameters = initialParameters;
		this.currentParameters = initialParameters;

		this.targetValues = targetValues;
		this.currentValues = new double[targetValues.length];

		this.finiteDifferenceStepSizes = finiteDifferenceStepSizes;
		this.weights = weights;

		this.targetAccuracy = targetAccuracy;
		this.maxNumberOfIterations = maxNumberOfIterations;
	}

	public GradientDescent(double[] initialParameters, double[] targetValues,double targetAccuracy, int maxNumberOfIterations) {
		this(initialParameters, targetValues, null, null, targetAccuracy, maxNumberOfIterations);

		this.weights = new double[targetValues.length];
		Arrays.fill(this.weights, 1.0 / (double)targetValues.length );
	}


	@Override
	public double[] getBestFitParameters() {
		return bestParameters;
	}

	@Override
	public double getRootMeanSquaredError() {
		return Math.sqrt(bestAccuracy);
	}

	@Override
	public int getIterations() {
		return numberOfIterations;
	}

	@Override
	public void run() throws SolverException {

		evaluateCurrentState();

		while(!isDone()){

			if(derivativeStorage == null){
				double[][] derivative = new double[currentParameters.length][targetValues.length];
				setDerivatives(currentParameters, derivative);
				
				derivativeStorage = new double[currentParameters.length];
				// take the weighted average for parameter update
				for(int parameterIndex = 0; parameterIndex < currentParameters.length; parameterIndex++){
					double averageDerivative = 0.0;
					for(int valueIndex = 0; valueIndex < targetValues.length; valueIndex++){
						averageDerivative += weights[valueIndex] * derivative[parameterIndex][valueIndex];
					}
					derivativeStorage[parameterIndex] = averageDerivative;
				}
			}

			for(int parameterIndex = 0; parameterIndex < currentParameters.length; parameterIndex++){
				currentParameters[parameterIndex] -= lambda * derivativeStorage[parameterIndex];
			}

			evaluateCurrentState();

//			System.out.println(numberOfIterations + "\t" + currentAccuracy);
			
			numberOfIterations++;
		}

	}

	@Override
	public abstract void setValues(double[] parameters, double[] values) throws SolverException;

	@Override
	public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
		// Calculate new derivatives. Note that this method is called only with
		// parameters = parameterCurrent, so we may use valueCurrent.

		Vector<Future<double[]>> valueFutures = new Vector<Future<double[]>>(currentParameters.length);
		for (int parameterIndex = 0; parameterIndex < currentParameters.length; parameterIndex++) {
			final double[] parametersNew	= parameters.clone();
			final double[] derivative		= derivatives[parameterIndex];

			final int workerParameterIndex = parameterIndex;
			Callable<double[]> worker = new  Callable<double[]>() {
				public double[] call() throws SolverException {
					double parameterFiniteDifference;
					if(finiteDifferenceStepSizes != null) {
						parameterFiniteDifference = finiteDifferenceStepSizes[workerParameterIndex];
					}
					else {
						/*
						 * Try to adaptively set a parameter shift. Note that in some
						 * applications it may be important to set parameterSteps.
						 * appropriately.
						 */
						parameterFiniteDifference = (Math.abs(parametersNew[workerParameterIndex]) + 1) * 1E-8;
					}

					// Shift parameter value
					parametersNew[workerParameterIndex] += parameterFiniteDifference;

					// Calculate derivative as (valueUpShift - valueCurrent) / parameterFiniteDifference
					try {
						setValues(parametersNew, derivative);
					} catch (Exception e) {
						// We signal an exception to calculate the derivative as NaN
						Arrays.fill(derivative, Double.NaN);
					}
					for (int valueIndex = 0; valueIndex < currentValues.length; valueIndex++) {
						derivative[valueIndex] -= currentValues[valueIndex];
						derivative[valueIndex] /= parameterFiniteDifference;
						if(Double.isNaN(derivative[valueIndex])) derivative[valueIndex] = 0.0;
					}
					return derivative;
				}
			};
			if(executor != null) {
				Future<double[]> valueFuture = executor.submit(worker);
				valueFutures.add(parameterIndex, valueFuture);
			}
			else {
				FutureTask<double[]> valueFutureTask = new FutureTask<double[]>(worker);
				valueFutureTask.run();
				valueFutures.add(parameterIndex, valueFutureTask);
			}
		}

		for (int parameterIndex = 0; parameterIndex < currentParameters.length; parameterIndex++) {
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

	private boolean isDone(){
		return currentAccuracy < targetAccuracy 
				||
				maxNumberOfIterations < numberOfIterations
				||
				lambda > maxLambda || lambda < minLambda
				||
				arrayEqualToDouble(derivativeStorage, 0.0)
				;
	}

	private boolean arrayEqualToDouble(double[] array, double a) {
		if(array == null) return false;
		for(double x : array) if(x != a) return false;
		return true;
	}

	private void evaluateCurrentState() throws SolverException{
		setValues(currentParameters, currentValues);

		double errorRMS = 0.0;
		for(int valueIndex = 0; valueIndex < currentValues.length; valueIndex++){
			double error = currentValues[valueIndex] - targetValues[valueIndex];
			errorRMS += weights[valueIndex] * (error * error);
		}
		currentAccuracy = Math.sqrt(errorRMS);

		if(currentAccuracy < bestAccuracy){
			bestAccuracy = currentAccuracy;
			bestParameters = currentParameters;

			lambda /= lambdaDivisor;

			// free cached derivative storage
			derivativeStorage = null;
		} else {
			lambda *= lambdaMultiplicator;
		}
	}

	public GradientDescent setLambda(double lambda, double lambdaDivisor, double lambdaMultiplicator){
		if(!isDone()){

			if(lambdaDivisor <= 1.0 || lambdaMultiplicator <= 1.0) 
				throw new IllegalArgumentException("Parameters lambdaDivisor and lambdaMultiplicator are required to be > 1.");

			this.lambda = lambda;
			this.lambdaDivisor = lambdaDivisor;
			this.lambdaMultiplicator = lambdaMultiplicator;			
		} else throw new UnsupportedOperationException("Solver cannot be modified after it has run.");

		return this;
	}
}


