package net.finmath.optimizer.gradientdescent;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import net.finmath.montecarlo.RandomVariable;
import net.finmath.optimizer.OptimizerInterface;
import net.finmath.optimizer.OptimizerInterfaceAAD;
import net.finmath.optimizer.SolverException;
import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;



/**
 * Simple Gradient Descent Optimizer
 * 
 * with varying step size:
 * <lu>
 * <li> if last step better than best result until now:
 * lambda<sub>k+1</sub> = lambda<sub>k</sub> / lambda<sub>Divisor</sub>,
 * </li>
 * <li> if last step worse than best result until now:
 * lambda<sub>k+1</sub>  = lambda<sub>k</sub>  * lambda<sub>Multiplicator</sub>,
 * </li>
 * </lu>
 * where lambda<sub>Divisor</sub> and lambda<sub>Divisor</sub> are greater than 1.0.
 * 
 * */
public abstract class GradientDescent implements Serializable, Cloneable, OptimizerInterface, DerivativeFunction {

	private static final long serialVersionUID = -84822697392025037L;

	private double targetAccuracy;
	private double currentAccuracy		= Double.POSITIVE_INFINITY;
	private double bestAccuracy 		= Double.POSITIVE_INFINITY;
	private int numberOfIterations		= 0;
	private int maxNumberOfIterations;


	private RealVector bestParameters;
	private RealVector currentParameters;

	private RealVector finiteDifferenceStepSizes;

	private RealVector targetValues;
	private RealVector currentValues;
	private RealVector weights;

	private double	lambda				= 1E-2;
	private double  minLambda			= 1E-10;
	private double  maxLambda			= 1E2;
	
	private double	lambdaDivisor		= 1.1;
	private double	lambdaMultiplicator	= 1.5;

	private RealVector derivativeStorageVector;

	private ExecutorService executor;

	public GradientDescent(double[] initialParameters, double[] targetValues, double[] weights, double[] finiteDifferenceStepSizes,
			double targetAccuracy, int maxNumberOfIterations) {
		this.bestParameters = new ArrayRealVector(initialParameters);
		this.currentParameters = new ArrayRealVector(initialParameters);
		
		this.targetValues = new ArrayRealVector(targetValues);
		this.currentValues = new ArrayRealVector(targetValues.length);

		this.finiteDifferenceStepSizes = finiteDifferenceStepSizes != null ? new ArrayRealVector(finiteDifferenceStepSizes) : null;
		int numberOfValues = targetValues.length;
		this.weights = finiteDifferenceStepSizes != null ? new ArrayRealVector(weights) : new ArrayRealVector(numberOfValues, 1.0 / numberOfValues);

		this.targetAccuracy = targetAccuracy;
		this.maxNumberOfIterations = maxNumberOfIterations;
		
		this.lambda *= (new RandomVariable(0.0, initialParameters)).abs().getAverage();
	}

	public GradientDescent(double[] initialParameters, double[] targetValues, double targetAccuracy, int maxNumberOfIterations) {
		this(initialParameters, targetValues, null, null, targetAccuracy, maxNumberOfIterations);
		int numberOfValues = targetValues.length;
		this.weights = new ArrayRealVector(numberOfValues, 1.0 / numberOfValues);
	}


	@Override
	public double[] getBestFitParameters() {
		return bestParameters.toArray();
	}

	@Override
	public double getRootMeanSquaredError() {
		return bestAccuracy;
	}

	@Override
	public int getIterations() {
		return numberOfIterations;
	}

	@Override
	public void run() throws SolverException {

		currentValues = getValues(currentParameters);
		currentAccuracy = currentValues.subtract(targetValues).getNorm();
		bestAccuracy = currentAccuracy;
		
		System.out.println("Step#: " +numberOfIterations + "\t currentAccuracy: " + currentAccuracy);
		
		RealVector newParameters;
		
		while(!isDone()){
			numberOfIterations++;
			
			if(derivativeStorageVector == null){
				RealMatrix derivativeMatrix = getDerivative(currentParameters);
				derivativeStorageVector = derivativeMatrix.operate(weights);
			}

			newParameters = currentParameters.subtract(derivativeStorageVector.mapMultiplyToSelf(lambda));
			currentValues = getValues(newParameters);

			currentParameters = newParameters.copy();
			currentAccuracy = currentValues.subtract(targetValues).getNorm();
					
			System.out.println("Step#: " + numberOfIterations + "\t currentAccuracy: " + currentAccuracy + "\t lambda=" + lambda);
	
			
			if(currentAccuracy < bestAccuracy){
				bestAccuracy = currentAccuracy;
				bestParameters = currentParameters.copy();
				
				lambda /= lambdaDivisor;
				// free cache
				derivativeStorageVector = null;
			} else {
				lambda *= lambdaMultiplicator;
			}
		}
	}
	
	private RealVector getValues(RealVector parameter) throws SolverException{
		double[] values = new double[targetValues.getDimension()];
		setValues(parameter.toArray(), values);
		return new ArrayRealVector(values);
	}
	
	private RealMatrix getDerivative(RealVector parameter) throws SolverException{
		double[][] derivative = new double[currentParameters.getDimension()][targetValues.getDimension()];
		setDerivatives(parameter.toArray(), derivative);
		return new Array2DRowRealMatrix(derivative, false /*copy array*/);
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
					double parameterFiniteDifference;
					if(finiteDifferenceStepSizes != null) {
						parameterFiniteDifference = finiteDifferenceStepSizes.getEntry(workerParameterIndex);
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
					for (int valueIndex = 0; valueIndex < currentValues.getDimension(); valueIndex++) {
						derivative[valueIndex] -= currentValues.getEntry(valueIndex);
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

	private boolean isDone(){
		return currentAccuracy < targetAccuracy 
				||
				maxNumberOfIterations < numberOfIterations
				||
				lambda > maxLambda || lambda < minLambda
//				||
//				derivativeStorageVector.getLInfNorm() == 0.0
				;
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


