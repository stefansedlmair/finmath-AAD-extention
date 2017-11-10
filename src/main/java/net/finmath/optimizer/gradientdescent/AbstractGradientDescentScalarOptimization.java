/**
 * 
 */
package net.finmath.optimizer.gradientdescent;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;

import net.finmath.functions.VectorAlgbra;
import net.finmath.optimizer.LevenbergMarquardt;
import net.finmath.optimizer.OptimizerInterface;
import net.finmath.optimizer.OptimizerInterfaceAAD;
import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;
import net.finmath.optimizer.SolverException;

/**
 * Abstract Class implementing the essentials of a Gradient descent algorithm for a scalar function <code>f</code>.
 * 
 * The update rule for the parameters under this algorithm is:
 * 
 * <center> x<sub>k+1</sub> = x<sub>k</sub> - stepSize * \nabla f(x<sub>k</sub>) </center>
 * 
 * 
 * @author Stefan Sedlmair
 * @version 1.0
 */
public abstract class AbstractGradientDescentScalarOptimization	implements Serializable, Cloneable, OptimizerInterfaceAAD, DerivativeFunction {


	private static final long serialVersionUID = -2563916306695472945L;

	protected double 	errorTolerance;
	protected double 	currentAccuracy				= Double.POSITIVE_INFINITY;
	protected double 	bestAccuracy 				= Double.POSITIVE_INFINITY;
	protected int		numberOfIterations			= 0;

	protected int 		maxNumberOfIterations;
	protected long		startTimeInMilliSeconds;
	protected long 		maxRunTimeInMilliSeconds;
	
	protected double[] finiteDifferenceStepSizes;

	protected double[] bestParameter;
	protected double[] currentParameter;

	protected double   currentValue;
	protected double   targetValue;

	protected boolean 	allowWorsening;

	protected ExecutorService 	executor;
	protected boolean			executorShutdownWhenDone;
	
	public AbstractGradientDescentScalarOptimization(double[] initialParameter, double targetValue, double errorTolerance, int maxNumberOfIterations, long 	maxRunTimeInMilliSeconds, double[] finiteDifferenceStepSizes, ExecutorService executor, boolean allowWorsening) {
		this.bestParameter = initialParameter;
		this.currentParameter = initialParameter;

		this.targetValue = targetValue;
		this.maxNumberOfIterations = maxNumberOfIterations;
		this.maxRunTimeInMilliSeconds = maxRunTimeInMilliSeconds;

		this.errorTolerance = errorTolerance;
		this.finiteDifferenceStepSizes = finiteDifferenceStepSizes;

		this.executor = executor;
		this.executorShutdownWhenDone = true;
		this.allowWorsening = allowWorsening;
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerInterface#getBestFitParameters()
	 */
	@Override
	public double[] getBestFitParameters() {
		return bestParameter;
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerInterface#getRootMeanSquaredError()
	 */
	@Override
	public double getRootMeanSquaredError() {
		return bestAccuracy;
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
		try {
			double stepSize = Double.NaN;

			// check accuracy with initial parameter set
			currentValue = getValue(currentParameter);
			currentAccuracy = Math.abs(currentValue - targetValue);
			bestAccuracy = currentAccuracy;

			startTimeInMilliSeconds = System.currentTimeMillis();
			
			while(!isDone()){
				// get step size for this gradient descent algorithm
				stepSize = getStepSize(currentParameter);

				// calculate the parameter update characteristic for gradient descent algorithms
				currentParameter = VectorAlgbra.subtract(currentParameter, VectorAlgbra.scalarProduct(stepSize, getDerivative(currentParameter)));

				// evaluate the new parameter set
				currentValue = getValue(currentParameter);

				// update accuracy and store last one to see advance
				double lastAccuracy = currentAccuracy;
				currentAccuracy = Math.abs(currentValue - targetValue);
				
				System.out.println(numberOfIterations + ";" + (System.currentTimeMillis() - startTimeInMilliSeconds) + ";" + currentAccuracy);
				
				// store best result
				if(currentAccuracy < bestAccuracy){			
					bestParameter = currentParameter.clone();
					bestAccuracy = currentAccuracy;

					// if advance is too slow => break
					if(lastAccuracy - currentAccuracy < errorTolerance)
						break;

				} 
				// in case the last update did not increase accuracy break if necessary.
				else if(!allowWorsening) break;

			}
		}
		finally {
			// Shutdown executor if present.
			if(executor != null && executorShutdownWhenDone) {
				executor.shutdown();
				executor = null;
			}
		}

	}

	protected abstract double getStepSize(double[] currentParameter) throws SolverException;

	// store last set of parameters and its value to avoid calculating the value twice, for the same parameter set.
	private double[] parameterStorageValue = null;
	private double   valueStorage;

	protected double getValue(double[] parameter) throws SolverException{
		if(!Arrays.equals(parameter, parameterStorageValue)){

			double[] values = new double[1 /*only one dimension*/];
			setValues(parameter, values);

			parameterStorageValue = parameter.clone();
			valueStorage = values[0];
		}
		return valueStorage;
	}

	// store last set of parameters and its derivative to avoid calculating the derivative twice, for the same parameter set.
	private double[] parameterStorageDerivative = null;
	private double[] derivativeStorage;	

	protected double[] getDerivative(double[] parameter) throws SolverException{
		if(!Arrays.equals(parameter, parameterStorageDerivative)) {
			double[][] derivative = new double[parameter.length][1 /*only one dimension*/];
			setDerivatives(parameter, derivative);

			parameterStorageDerivative = parameter.clone();
			derivativeStorage = VectorAlgbra.transpose(derivative)[0];
		}

		return derivativeStorage;
	}

	protected boolean isDone(){
		return 	(maxNumberOfIterations <= getIterations())
				||
				(maxRunTimeInMilliSeconds <= (System.currentTimeMillis() - startTimeInMilliSeconds))
				||
				/*if optimizer is not allowed to get worse over time currentAccuracy always has to be equal to bestAccuracy */
				(allowWorsening ? false : currentAccuracy != bestAccuracy)
				||
				Double.isNaN(currentValue)
				||
				VectorAlgbra.containsNaN(currentParameter)
				;
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerInterface.ObjectiveFunction#setValues(double[], double[])
	 */
	@Override
	public abstract void setValues(double[] parameters, double[] values) throws SolverException;

	/**
	 * The derivative of the objective function. You may override this method
	 * if you like to implement your own derivative.
	 * 
	 * @param parameters Input value. The parameter vector.
	 * @param derivatives Output value, where derivatives[i][j] is d(value(j)) / d(parameters(i)
	 * @throws SolverException Thrown if the valuation fails, specific cause may be available via the <code>cause()</code> method.
	 * 
	 * @see LevenbergMarquardt
	 */
	public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
		// Calculate new derivatives. Note that this method is called only with
		// parameters = parameterCurrent, so we may use valueCurrent.

		Vector<Future<double[]>> valueFutures = new Vector<Future<double[]>>(currentParameter.length);
		for (int parameterIndex = 0; parameterIndex < currentParameter.length; parameterIndex++) {
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
					for (int valueIndex = 0; valueIndex < 1; valueIndex++) {
						derivative[valueIndex] -= currentValue;
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

		for (int parameterIndex = 0; parameterIndex < currentParameter.length; parameterIndex++) {
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

	public abstract OptimizerInterface cloneWithModifiedParameters(Map<String, Object> properties);
	
	protected void setProperties(Map<String, Object> properties, AbstractGradientDescentScalarOptimization cloneFather){
		this.targetValue 				= (double) 		properties.getOrDefault("targetValue",				cloneFather.targetValue);
		this.errorTolerance 			= (double) 		properties.getOrDefault("errorTolerance",			cloneFather.errorTolerance);
		this.maxNumberOfIterations		= (int) 		properties.getOrDefault("maxNumberOfIterations", 	cloneFather.maxNumberOfIterations);
		this.maxRunTimeInMilliSeconds 	= (long) 		properties.getOrDefault("maxRunTimeInMillis", 		cloneFather.maxRunTimeInMilliSeconds);
		this.finiteDifferenceStepSizes 	= (double[]) 	properties.getOrDefault("finiteDifferenceStepSizes", cloneFather.finiteDifferenceStepSizes);
		double[] initialParameter 		= (double[]) 	properties.getOrDefault("initialParameters", 		cloneFather.currentParameter);
		this.currentParameter 			= initialParameter.clone();
		this.bestParameter 				= initialParameter.clone();
		this.executorShutdownWhenDone	= (boolean) 	properties.getOrDefault("executorShutdownWhenDone",	cloneFather.executorShutdownWhenDone);
		this.allowWorsening 			= (boolean) 	properties.getOrDefault("allowWorsening",			cloneFather.allowWorsening);
		this.executor 					= (ExecutorService) properties.getOrDefault("executor",				cloneFather.executor);
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException {
		return cloneWithModifiedParameters(new HashMap<>());
	}
}
