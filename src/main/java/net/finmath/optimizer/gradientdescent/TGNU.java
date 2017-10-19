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
public abstract class TGNU implements Serializable, Cloneable, OptimizerInterface, DerivativeFunction {

	private static final long serialVersionUID = -84822697392025037L;

	private double targetAccuracy;
	private double minAdvancePerStep;
	
	private double currentAccuracy		= Double.POSITIVE_INFINITY;
	private double bestAccuracy 		= Double.POSITIVE_INFINITY;
	private int numberOfIterations		= 0;
	private int maxNumberOfIterations;


	private RealVector bestParameters;
	private RealVector currentParameters;

	private double finiteDifferenceStepSize;

	private double   currentValue;
	private double   targetValue;

	RealVector derivativeStorageVector;

	private ExecutorService executor;

	public TGNU(double[] initialParameters, double targetValue, double finiteDifferenceStepSize, double targetAccuracy, double minAdvancePerStep ,int maxNumberOfIterations) {
		this.bestParameters = new ArrayRealVector(initialParameters);
		this.currentParameters = new ArrayRealVector(initialParameters);

		this.targetValue = targetValue;
		this.minAdvancePerStep = minAdvancePerStep;
		
		this.finiteDifferenceStepSize = finiteDifferenceStepSize;

		this.targetAccuracy = targetAccuracy;
		this.maxNumberOfIterations = maxNumberOfIterations;
	}

	public TGNU(double[] initialParameters, double targetValue, double minAdvancePerStep, double targetAccuracy, int maxNumberOfIterations) {
		this(initialParameters, targetValue, Double.NaN, targetAccuracy,minAdvancePerStep ,maxNumberOfIterations);
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
		
		currentValue = getValue(currentParameters);
		currentAccuracy = Math.abs(currentValue - targetValue);
		bestAccuracy = currentAccuracy;
		numberOfIterations++;

		System.out.println("Step#: " +numberOfIterations + "\t currentAccuracy: " + currentAccuracy);
		
		while(!isDone()){

			RealMatrix derivativeMatrix = getDerivative(currentParameters);
			derivativeStorageVector = derivativeMatrix.getColumnVector(0);

			double gradientNormSquare = derivativeStorageVector.dotProduct(derivativeStorageVector);

//			if(		getIterations() < 2) 	gradientNormSquare *= 1E-3;
//			else if(getIterations() < 5) 	gradientNormSquare *= 5E-2;
//			else if(getIterations() < 8) 	gradientNormSquare *= 1E-1;
//			else							gradientNormSquare *= 1E-1;
			
			RealVector newParameters = currentParameters.subtract(derivativeStorageVector.mapMultiply(currentValue/gradientNormSquare));
			double newValue = getValue(newParameters);
			
			currentParameters = newParameters.copy();
			currentValue = newValue;
			currentAccuracy = Math.abs(currentValue - targetValue);
					
			System.out.println("Step#: " +numberOfIterations + "\t currentAccuracy: " + currentAccuracy);
//			System.out.println(currentParameters);
				
			if(currentAccuracy < bestAccuracy){
//				double bestAccuracy = this.bestAccuracy;
				this.bestAccuracy = currentAccuracy;
				bestParameters = currentParameters.copy();
//				if(Math.abs(bestAccuracy - currentAccuracy) < minAdvancePerStep) break;
			}
//			else break;
	
			numberOfIterations++;
		}

	}
	
	private double getValue(RealVector parameter) throws SolverException{
		double[] values = new double[1];
		setValues(parameter.toArray(), values);
		return values[0];
	}
	
	private RealMatrix getDerivative(RealVector parameter) throws SolverException{
		double[][] derivative = new double[currentParameters.getDimension()][1 /*only one dimension*/];
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
					if(!Double.isNaN(finiteDifferenceStepSize)) {
						parameterFiniteDifference = finiteDifferenceStepSize;
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

					derivative[0] -= currentValue;
					derivative[0] /= parameterFiniteDifference;
					if(Double.isNaN(derivative[0])) derivative[0] = 0.0;

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
				;
	}
}


