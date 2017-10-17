/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 20.05.2006
 */
package net.finmath.montecarlo.interestrate.modelplugins;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.logging.Level;
import java.util.logging.Logger;

import net.finmath.exception.CalculationException;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.BrownianMotionInterface;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.montecarlo.automaticdifferentiation.forward.RandomVariableADFactory.RandomVariableAD;
import net.finmath.montecarlo.interestrate.LIBORMarketModelInterface;
import net.finmath.montecarlo.interestrate.LIBORModelMonteCarloSimulation;
import net.finmath.montecarlo.interestrate.products.AbstractLIBORMonteCarloProduct;
import net.finmath.montecarlo.process.ProcessEulerScheme;
import net.finmath.montecarlo.process.ProcessEulerScheme.Scheme;
import net.finmath.optimizer.OptimizerFactoryInterface;
import net.finmath.optimizer.OptimizerFactoryLevenbergMarquardt;
import net.finmath.optimizer.OptimizerInterface;
import net.finmath.optimizer.OptimizerInterface.ObjectiveFunction;
import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;
import net.finmath.optimizer.SolverException;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretizationInterface;

/**
 * Base class for parametric covariance models, see also {@link AbstractLIBORCovarianceModel}.
 * 
 * Parametric models feature a parameter vector which can be inspected
 * and modified for calibration purposes.
 * 
 * The parameter vector may have zero length, which indicated that the model
 * is not calibrateable.
 * 
 * This class includes the implementation of a generic calibration algorithm.
 * If you provide an arbitrary list of calibration products, the class can return
 * a new instance where the parameters are chosen such that the (weighted) root-mean-square 
 * error of the difference of the value of the calibration products and given target
 * values is minimized.
 * 
 * @author Christian Fries
 * @date 20.05.2006
 * @date 23.02.2014
 * @version 1.1
 */
public abstract class AbstractLIBORCovarianceModelParametric extends AbstractLIBORCovarianceModel {

	public enum OptimizerDerivativeType{
		FINITE_DIFFERENCES, ADJOINT_ALGORITHMIC_DIFFERENCIATION, ALGORITHMIC_DIFFERENCIATION
	}
	
	public enum OptimizerSolverType{
		VECTOR, SKALAR
	}

	private static final Logger logger = Logger.getLogger("net.finmath");
	private OptimizerInterface optimizer = null;

	/**
	 * Constructor consuming time discretizations, which are handled by the super class.
	 * 
	 * @param timeDiscretization The vector of simulation time discretization points.
	 * @param liborPeriodDiscretization The vector of tenor discretization points.
	 * @param numberOfFactors The number of factors to use (a factor reduction is performed)
	 */
	public AbstractLIBORCovarianceModelParametric(TimeDiscretizationInterface timeDiscretization, TimeDiscretizationInterface liborPeriodDiscretization, int numberOfFactors) {
		super(timeDiscretization, liborPeriodDiscretization, numberOfFactors);
	}

	/**
	 * Get the parameters of determining this parametric
	 * covariance model. The parameters are usually free parameters
	 * which may be used in calibration.
	 * 
	 * @return Parameter vector.
	 */
	
	public abstract RandomVariableInterface[] getParameterAsRandomVariable();
	
	public double[]	getParameter() {
		// get parameters
		RandomVariableInterface[] parameterAsRandomVariable = getParameterAsRandomVariable();

		// cover case of not calibrateable models
		if(parameterAsRandomVariable == null) return null;

		// get values of deterministic random variables
		double[] parameter = new double[parameterAsRandomVariable.length];
		for(int parameterIndex = 0; parameterIndex < parameterAsRandomVariable.length; parameterIndex++)
			parameter[parameterIndex] = parameterAsRandomVariable[parameterIndex].get(0);

		return parameter;
	}
	
	public long[]		getParameterID() {
		RandomVariableInterface[] parameterAsRandomVariable = getParameterAsRandomVariable();
		
		if(parameterAsRandomVariable == null || !(parameterAsRandomVariable[0] instanceof RandomVariableDifferentiableInterface)) return null;
		
		long[] parameterIDs = new long[parameterAsRandomVariable.length];
		for(int parameterIndex = 0; parameterIndex < parameterIDs.length; parameterIndex++)
			parameterIDs[parameterIndex] = ((RandomVariableDifferentiableInterface) parameterAsRandomVariable[parameterIndex]).getID();
		
		return parameterIDs;
	}

	@Override
	public abstract Object clone();

	/**
	 * Return an instance of this model using a new set of parameters.
	 * Note: To improve performance it is admissible to return the same instance of the object given that the parameters have not changed. Models should be immutable.
	 * 
	 * @param parameters The new set of parameters.
	 * @return An instance of AbstractLIBORCovarianceModelParametric with modified parameters.
	 */
	public abstract AbstractLIBORCovarianceModelParametric getCloneWithModifiedParameters(double[] parameters);

	public AbstractLIBORCovarianceModelParametric getCloneCalibrated(final LIBORMarketModelInterface calibrationModel, final AbstractLIBORMonteCarloProduct[] calibrationProducts, double[] calibrationTargetValues, double[] calibrationWeights) throws CalculationException {
		return getCloneCalibrated(calibrationModel, calibrationProducts, calibrationTargetValues, calibrationWeights, null);
	}

	/**
	 * Performs a generic calibration of the parametric model by
	 * trying to match a given vector of calibration product to a given vector of target values
	 * using a given vector of weights.
	 * 
	 * Optional calibration parameters may be passed using the map calibrationParameters. The keys are (<code>String</code>s):
	 * <ul>
	 * 	<li><tt>brownianMotion</tt>: Under this key an object implementing {@link net.finmath.montecarlo.BrownianMotionInterface} may be provided. If so, this Brownian motion is used to build the valuation model.</li>
	 * 	<li><tt>maxIterations</tt>: Under this key an object of type Integer may be provided specifying the maximum number of iterations.</li>
	 * 	<li><tt>accuracy</tt>: Under this key an object of type Double may be provided specifying the desired accuracy. Note that this is understood in the sense that the solver will stop if the iteration does not improve by more than this number.</li>
	 * </ul>
	 * 
	 * @param calibrationModel The LIBOR market model to be used for calibrations (specifies forward curve and tenor discretization).
	 * @param calibrationProducts The array of calibration products.
	 * @param calibrationTargetValues The array of target values.
	 * @param calibrationWeights The array of weights.
	 * @param calibrationParameters A map of type Map&lt;String, Object&gt; specifying some (optional) calibration parameters.
	 * @return A new parametric model of the same type than <code>this</code> one, but with calibrated parameters.
	 * @throws CalculationException Thrown if calibration has failed.
	 */
	public AbstractLIBORCovarianceModelParametric getCloneCalibrated(final LIBORMarketModelInterface calibrationModel, final AbstractLIBORMonteCarloProduct[] calibrationProducts, final double[] calibrationTargetValues, double[] calibrationWeights, Map<String,Object> calibrationParameters) throws CalculationException {
		
		/*
		 * We allow for 2 simultaneous calibration models.
		 * Note: In the case of a Monte-Carlo calibration, the memory requirement is that of
		 * one model with 2 times the number of paths. In the case of an analytic calibration
		 * memory requirement is not the limiting factor.
		 */
		final int numberOfThreads = 2;
		final ExecutorService executor = null;
		
		//int numberOfThreadsForProductValuation = 2 * Math.max(2, Runtime.getRuntime().availableProcessors());
		//Executors.newFixedThreadPool(numberOfThreads);//numberOfThreadsForProductValuation);
		
		double[] initialParameters = this.getParameter();
				
		int numberOfCalibrationProducts = calibrationProducts.length;
		int numberOfParameters = initialParameters.length;
		
		if(numberOfCalibrationProducts != calibrationTargetValues.length) throw new IllegalArgumentException("Each calibration product has to have a target value!");
		
		// get calibration parameters
		if(calibrationParameters == null) calibrationParameters = new HashMap<String,Object>();
		
		final int 						numberOfPaths		= (int)calibrationParameters.getOrDefault(		"numberOfPaths", 		2000);
		final int 						seed				= (int)calibrationParameters.getOrDefault(		"seed", 				31415);
		final int 						maxIterations		= (int)calibrationParameters.getOrDefault(		"maxIterations", 		400);
		final double					parameterStepValue	= (double)calibrationParameters.getOrDefault(	"parameterStep", 		1E-4);
		final double					accuracy			= (double)calibrationParameters.getOrDefault(	"accuracy", 			1E-7);
		final Scheme 					processScheme		= (Scheme)calibrationParameters.getOrDefault(	"scheme",  				Scheme.EULER_FUNCTIONAL);
		final double[] 					lowerBound 			= (double[])calibrationParameters.getOrDefault(	"parameterLowerBound", 	initialzeDoubleArray(Double.NEGATIVE_INFINITY, numberOfParameters));
		final double[] 					upperBound 			= (double[])calibrationParameters.getOrDefault(	"parameterUpperBound", 	initialzeDoubleArray(Double.POSITIVE_INFINITY, numberOfParameters));
		final OptimizerSolverType 		solverType 			= (OptimizerSolverType) calibrationParameters.getOrDefault(		"solverType", 			OptimizerSolverType.VECTOR);
		final OptimizerDerivativeType	derivativeType 		= (OptimizerDerivativeType) calibrationParameters.getOrDefault(	"derivativeType", 		OptimizerDerivativeType.FINITE_DIFFERENCES);
		final BrownianMotionInterface 	brownianMotion		= (BrownianMotionInterface)calibrationParameters.getOrDefault(	"brownianMotion", 	new BrownianMotion(getTimeDiscretization(), getNumberOfFactors(), numberOfPaths, seed));
		final OptimizerFactoryInterface optimizerFactory 	= (OptimizerFactoryInterface)calibrationParameters.getOrDefault("optimizerFactory", new OptimizerFactoryLevenbergMarquardt(maxIterations, accuracy, numberOfThreads));

		final double[] parameterStep = initialzeDoubleArray(parameterStepValue, numberOfParameters);
		

		// define evaluating functions
		ObjectiveFunction calibrationError = new ObjectiveFunction() {			
			// Calculate model values for given parameters
			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {

				AbstractLIBORCovarianceModelParametric calibrationCovarianceModel = AbstractLIBORCovarianceModelParametric.this.getCloneWithModifiedParameters(parameters);
				
				ArrayList<Future<RandomVariableInterface>> valueFutures = 
						getFutureValuesFromParameters(calibrationCovarianceModel, calibrationModel, brownianMotion, calibrationProducts, 
														numberOfCalibrationProducts, executor, calibrationTargetValues, processScheme);
				
				// get calculated prices
				double[] calibratedPrices = new double[numberOfCalibrationProducts];
				for(int calibrationProductIndex=0; calibrationProductIndex < numberOfCalibrationProducts; calibrationProductIndex++)
					try { calibratedPrices[calibrationProductIndex] = valueFutures.get(calibrationProductIndex).get().getAverage(); }
					catch (InterruptedException | ExecutionException e) { throw new SolverException(e);}
				
				switch(solverType) {
				case VECTOR:
					// copy the prices under the current calibration directly to the values
					System.arraycopy(calibratedPrices, 0, values, 0, numberOfCalibrationProducts);
					break;
				case SKALAR:
					// calculate the mean-square-error
					double errorRMS = 0.0;
					for(int i = 0; i < numberOfCalibrationProducts; i++) {
							double error = calibratedPrices[i] - calibrationTargetValues[i];
							errorRMS += error * error;
					}
//					errorRMS /= (double) numberOfCalibrationProducts;
					System.arraycopy(new double[] {Math.sqrt(errorRMS)} , 0, values, 0, 1);
					break;
				}
			}		
		};
		
		DerivativeFunction calibrationErrorExtended = new DerivativeFunction() {
			
			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
				calibrationError.setValues(parameters, values);
			}
			
			@Override
			public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
				
				long startTime = System.currentTimeMillis();
				
				AbstractLIBORCovarianceModelParametric calibrationCovarianceModel = AbstractLIBORCovarianceModelParametric.this.getCloneWithModifiedParameters(parameters);
				
				ArrayList<Future<RandomVariableInterface>> valueFutures = 
						getFutureValuesFromParameters(calibrationCovarianceModel, calibrationModel, brownianMotion, calibrationProducts, 
														numberOfCalibrationProducts, executor, calibrationTargetValues, processScheme);
				
				// get calculated prices
				RandomVariableInterface[] calibratedPrices = new RandomVariableInterface[numberOfCalibrationProducts];
				for(int calibrationProductIndex=0; calibrationProductIndex < numberOfCalibrationProducts; calibrationProductIndex++)
					try { calibratedPrices[calibrationProductIndex] = valueFutures.get(calibrationProductIndex).get(); }
					catch (InterruptedException | ExecutionException e) { throw new SolverException(e);}

				// for convenience
				RandomVariableInterface zero = calibrationModel.getRandomVariableForConstant(0.0);

				switch(solverType) {
				case VECTOR:
					switch(derivativeType) {
					case ADJOINT_ALGORITHMIC_DIFFERENCIATION:
						for(int productIndex=0; productIndex < numberOfCalibrationProducts; productIndex++) {
							RandomVariableInterface calibratedPrice = calibratedPrices[productIndex];

							Map<Long, RandomVariableInterface> gradient = ((RandomVariableDifferentiableInterface) calibratedPrice).getGradient();
						
							// request the ids of the parameters from the calibrated model
							long[] keys = calibrationCovarianceModel.getParameterID();					
							for(int parameterIndex = 0; parameterIndex < parameters.length; parameterIndex++) 
								// do not stop the optimizer when derivative is not found. Set default to zero.
								derivatives[parameterIndex][productIndex] = gradient.getOrDefault(keys[parameterIndex], zero).getAverage();
						}	
						break;
					case ALGORITHMIC_DIFFERENCIATION:
						RandomVariableInterface[] parameterRandomVariables = calibrationCovarianceModel.getParameterAsRandomVariable();
						
						for(int parameterIndex = 0; parameterIndex < parameters.length; parameterIndex++) {
							Map<Long, RandomVariableInterface> partialDerivatives = ((RandomVariableAD) parameterRandomVariables[parameterIndex]).getAllPartialDerivatives();
							for(int productIndex = 0; productIndex < numberOfCalibrationProducts; productIndex++) {
								long productID = ((RandomVariableDifferentiableInterface) calibratedPrices[productIndex]).getID();
								derivatives[parameterIndex][productIndex] = partialDerivatives.getOrDefault(productID, zero).getAverage();
							}
						}
						break;
					}
					break;
				case SKALAR:
					// get the root-mean-square error of the calibration for every path
					RandomVariableInterface errorRMS = null;
					for(int i = 0; i < numberOfCalibrationProducts; i++) {
							RandomVariableInterface error = calibratedPrices[i].sub(calibrationTargetValues[i]);
							errorRMS = (errorRMS == null) ? error.squared() : errorRMS.addProduct(error, error);
					}
//					errorRMS = errorRMS.div(numberOfCalibrationProducts).sqrt();
					errorRMS = errorRMS.sqrt();
							
					// take gradient of the mean-square-error (here AAD should bring the most improvement!)
					Map<Long, RandomVariableInterface> gradient = ((RandomVariableDifferentiableInterface) errorRMS).getGradient();
										
					// request the ids of the parameters from the calibrated model
					long[] keys = calibrationCovarianceModel.getParameterID();
					
					// fill in the calculated gradient in the derivative matrix
					for(int parameterIndex = 0; parameterIndex < parameters.length; parameterIndex++) 
						// do not stop the optimizer when derivative is not found. Set default to zero.
						derivatives[parameterIndex][0] = gradient.getOrDefault(keys[parameterIndex], zero).getAverage();
					break;
				}
				long endTime = System.currentTimeMillis();

				System.out.println("calculation time for derivative (AD): " + (endTime - startTime)/1E3 + "s"); 
				System.out.println();
			}
		};
		
		// in case of a skalar solving the target value will be zero
		double[] targetValues = null;
		switch(solverType) {
		case VECTOR:
//			targetValues = calibrationTargetValues;
			targetValues = new double[numberOfCalibrationProducts];
			System.arraycopy(calibrationTargetValues, 0, targetValues, 0, numberOfCalibrationProducts);
			break;
		case SKALAR:
			targetValues = new double[] {0.0};
			break;
		}

//		OptimizerInterface optimizer = null;
		switch(derivativeType) {
		case FINITE_DIFFERENCES:
			optimizer = optimizerFactory.getOptimizer(calibrationError, initialParameters, lowerBound, upperBound, parameterStep, targetValues);
			break;
		case ADJOINT_ALGORITHMIC_DIFFERENCIATION:
		case ALGORITHMIC_DIFFERENCIATION:
			optimizer = optimizerFactory.getOptimizer(calibrationErrorExtended, initialParameters, lowerBound, upperBound, parameterStep, targetValues);
			break;
		}
					
		try {
			optimizer.run();
		}
		catch(SolverException e) {
			throw new CalculationException(e);
		}
		finally {
			if(executor != null) {
				executor.shutdown();
			}
		}

		// Get covariance model corresponding to the best parameter set.
		double[] bestParameters = optimizer.getBestFitParameters();
		AbstractLIBORCovarianceModelParametric calibrationCovarianceModel = this.getCloneWithModifiedParameters(bestParameters);

		// Diagnostic output
		if (logger.isLoggable(Level.FINE)) {
			logger.fine("The solver required " + optimizer.getIterations() + " iterations. The best fit parameters are:");

			String logString = "Best parameters:";
			for(int i=0; i<bestParameters.length; i++) {
				logString += "\tparameter["+i+"]: " + bestParameters[i];
			}
			logger.fine(logString);
		}

		return calibrationCovarianceModel;    	
	}

	private ArrayList<Future<RandomVariableInterface>> getFutureValuesFromParameters(AbstractLIBORCovarianceModel calibrationCovarianceModel, LIBORMarketModelInterface calibrationModel, BrownianMotionInterface brownianMotion, final AbstractLIBORMonteCarloProduct[] calibrationProducts, int numberOfCalibrationProducts, ExecutorService executor, double[] calibrationTargetValues, Scheme processScheme){
				
		// Create a LIBOR market model with the new covariance structure.
		LIBORMarketModelInterface model = calibrationModel.getCloneWithModifiedCovarianceModel(calibrationCovarianceModel);
		ProcessEulerScheme process = new ProcessEulerScheme(brownianMotion, processScheme);
		final LIBORModelMonteCarloSimulation liborMarketModelMonteCarloSimulation =  new LIBORModelMonteCarloSimulation(model, process);

		ArrayList<Future<RandomVariableInterface>> valueFutures = new ArrayList<Future<RandomVariableInterface>>(numberOfCalibrationProducts);
		for(int calibrationProductIndex=0; calibrationProductIndex<numberOfCalibrationProducts; calibrationProductIndex++) {
			final int workerCalibrationProductIndex = calibrationProductIndex;
			Callable<RandomVariableInterface> worker = new  Callable<RandomVariableInterface>() {
				public RandomVariableInterface call() throws SolverException {
					try {
						return calibrationProducts[workerCalibrationProductIndex].getValue(0.0, liborMarketModelMonteCarloSimulation);
					} catch (CalculationException e) {
						// We do not signal exceptions to keep the solver working and automatically exclude non-working calibration products.
//						System.out.println("no value for: " + calibrationProducts[workerCalibrationProductIndex]);
						return  calibrationModel.getRandomVariableForConstant(calibrationTargetValues[workerCalibrationProductIndex]); 
//						return new RandomVariable(0.0, calibrationTargetValues[workerCalibrationProductIndex]);
					} catch (Exception e) {
						// We do not signal exceptions to keep the solver working and automatically exclude non-working calibration products.
//						System.out.println("no value for: " + calibrationProducts[workerCalibrationProductIndex]);
						return  calibrationModel.getRandomVariableForConstant(calibrationTargetValues[workerCalibrationProductIndex]); 
//						return new RandomVariable(0.0, calibrationTargetValues[workerCalibrationProductIndex]);
					}
				}
			};
			if(executor != null) {
				Future<RandomVariableInterface> valueFuture = executor.submit(worker);
				valueFutures.add(calibrationProductIndex, valueFuture);
			}
			else {
				FutureTask<RandomVariableInterface> valueFutureTask = new FutureTask<RandomVariableInterface>(worker);
				valueFutureTask.run();
				valueFutures.add(calibrationProductIndex, valueFutureTask);
			}
		}
		return valueFutures;
	}
	
	public OptimizerInterface getCalibrationOptimizer() {
		return optimizer; 
	}
	
	
	@Override
	public String toString() {
		return "AbstractLIBORCovarianceModelParametric [getParameter()="
				+ Arrays.toString(getParameter()) + "]";
	}
	
	private double[] initialzeDoubleArray(double value, int length) {
		double[] array = new double[length];
		Arrays.fill(array, value);
		return array;
	}
}
