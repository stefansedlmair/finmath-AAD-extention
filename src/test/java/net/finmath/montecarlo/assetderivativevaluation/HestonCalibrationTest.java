package net.finmath.montecarlo.assetderivativevaluation;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;

import net.finmath.exception.CalculationException;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.IndependentIncrementsInterface;
import net.finmath.montecarlo.assetderivativevaluation.HestonModel.Scheme;
import net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.montecarlo.automaticdifferentiation.backward.RandomVariableDifferentiableAADFactory;
import net.finmath.montecarlo.process.ProcessEulerScheme;
import net.finmath.optimizer.LevenbergMarquardt;
import net.finmath.optimizer.SolverException;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretization.ShortPeriodLocation;
import net.finmath.time.TimeDiscretizationInterface;

public class HestonCalibrationTest {

	private static DecimalFormat formatterValue		= new DecimalFormat(" ##0.000%;-##0.000%", new DecimalFormatSymbols(Locale.ENGLISH));
	private static DecimalFormat formatterDeviation	= new DecimalFormat(" 0.00000E00;-0.00000E00", new DecimalFormatSymbols(Locale.ENGLISH));

	AbstractRandomVariableFactory randomVariableFactory;

	LevenbergMarquardt optimizer;
	int numberOfIterations 	= 100;
	double riskFreeRate 	= 0.05;
	int numberOfThreads		= 2;

	double delta = 1E-2;	

	double[] strikes = new double[]{60, 80, 100, 105, 110, 120, 140, 160, 180};
	double[] targetImplBSVolatilities = new double[]{56.51/100.0, 52.47/100.0, 50.90/100.0, 50.75/100.0, 50.85/100.0, 51.17/100.0, 52.55/100.0, 54.40/100.0, 56.05/100.0};
	EuropeanOption[] options;
	double[] targetPrices;
	
	double initialValue = 100.0;
	double maturity 	= 1.0;

	ProcessEulerScheme 	process;

	public HestonCalibrationTest() {

		int seed = 1234;
		int numberOfPaths 	= (int) 5E4;
		int numberOfFactors = 2;

		double startTime 	= 0.0;
		double endTime 		= 1.0;
		double deltaT		= 0.5;

		TimeDiscretizationInterface timeDiscretization = new TimeDiscretization(startTime, endTime, deltaT, ShortPeriodLocation.SHORT_PERIOD_AT_END);

		IndependentIncrementsInterface brownianMotion = new BrownianMotion(timeDiscretization, numberOfFactors, numberOfPaths, seed);
		process = new ProcessEulerScheme(brownianMotion);

		options = new EuropeanOption[strikes.length];
		for(int optionIndex = 0; optionIndex < strikes.length; optionIndex++)
			options[optionIndex] = new EuropeanOption(maturity, strikes[optionIndex]);

		targetPrices = new double[targetImplBSVolatilities.length];
		for(int i=0; i < targetPrices.length; i++)
			targetPrices[i] = AnalyticFormulas.blackScholesDigitalOptionValue(initialValue, riskFreeRate, targetImplBSVolatilities[i], maturity, strikes[i]);
		
		randomVariableFactory = new RandomVariableDifferentiableAADFactory();
	}

	@Test
	public void HestonModel() throws CalculationException, SolverException{

		// Heston Settings
		double initialVolatility	= 0.3;
		double intitalDiscountRate	= 0.05;
		double intitialKappa 		= 0.2;
		double intitialTheta 		= 0.3;
		double intitialXi			= 0.2;
		double intitialRho			= 0.0;

		double[] initialParameters = new double[]{initialVolatility, intitalDiscountRate, intitialKappa, intitialTheta, intitialXi, intitialRho};

		// define optimizer
		optimizer = new LevenbergMarquardt(initialParameters, targetPrices, numberOfIterations, numberOfThreads) {

			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {

				// Heston Settings
				double volatility 		= parameters[0];
				double discountRate		= parameters[1];
				double kappa 			= parameters[2];
				double theta 			= parameters[3];
				double xi				= parameters[4];
				double rho				= parameters[5];

				HestonModel hestonModel = new HestonModel(initialValue, riskFreeRate, volatility, discountRate, theta, kappa, xi, rho, Scheme.FULL_TRUNCATION, randomVariableFactory);

				MonteCarloAssetModel mcAssetModel = new MonteCarloAssetModel(hestonModel, process.clone());

				for(int optionIndex = 0; optionIndex < options.length; optionIndex++){
					try {
						EuropeanOption option	= options[optionIndex];

						RandomVariableInterface optionValues = option.getValue(0.0, mcAssetModel);
						double optionPrice = optionValues.getAverage();

						values[optionIndex] = optionPrice;
					} catch (CalculationException e) {}

				}
			}

			@Override
			public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
			
				// Heston Settings
				double volatility 		= parameters[0];
				double discountRate		= parameters[1];
				double kappa 			= parameters[2];
				double theta 			= parameters[3];
				double xi				= parameters[4];
				double rho				= parameters[5];

				HestonModel hestonModel = new HestonModel(initialValue, riskFreeRate, volatility, discountRate, theta, kappa, xi, rho, Scheme.FULL_TRUNCATION, randomVariableFactory);

				// estimate the ids of the parameters
				long volatilityID		= ((RandomVariableDifferentiableInterface) hestonModel.getRiskFreeRate()).getID();
				long discountRateID		= volatilityID 		+ 1;
				long thetaID			= discountRateID 	+ 1;
				long kappaID			= thetaID 			+ 1;
				long xiID				= kappaID 			+ 1;
				long rhoID				= xiID 				+ 1;
				
				long[] parameterIDs = new long[] {volatilityID, discountRateID, thetaID, kappaID, xiID, rhoID};
				
				MonteCarloAssetModel mcAssetModel = new MonteCarloAssetModel(hestonModel, process.clone());

				for(int optionIndex = 0; optionIndex < options.length; optionIndex++){
					try {
						EuropeanOption option	= options[optionIndex];

						RandomVariableInterface optionValues = option.getValue(0.0, mcAssetModel);

						Map<Long, RandomVariableInterface> gradient = ((RandomVariableDifferentiableInterface) optionValues).getGradient();
						
						for(int parameterIndex = 0; parameterIndex < parameters.length; parameterIndex++) {
							long id = parameterIDs[parameterIndex];
							
							if(!gradient.containsKey(id)) throw new UnknownError("ID not found in gradient");
							
							double derivative = gradient.get(id).getAverage();
							
//							System.out.println(derivatives[optionIndex][parameterIndex]);
							if(Double.isNaN(derivative)) throw new UnknownError("Derivative for parameter " + parameterIndex + " in Option with Strike " + strikes[optionIndex] + " is Not-a-Number!");
							
							derivatives[optionIndex][parameterIndex] = derivative;; 

							
						}
						
					} catch (Exception e) {
						// TODO: handle exception
					}
				}



			}
		};

		// run optimizer
		optimizer.run();

		// print best parameters 
		System.out.println("Calibrated Heston Model");	
		double[] parameters = optimizer.getBestFitParameters();
		System.out.println("intitial volatility..." + formatterValue.format(parameters[0]));
		System.out.println("discount rate........." + formatterValue.format(parameters[1]));
		System.out.println("kappa................." + formatterValue.format(parameters[2]));
		System.out.println("theta................." + formatterValue.format(parameters[3]));
		System.out.println("xi...................." + formatterValue.format(parameters[4]));
		System.out.println("rho..................." + formatterValue.format(parameters[5]));

	}

	@After
	public void printResults(){
		double[] bestParameters = optimizer.getBestFitParameters();
		double[] finalValues = new double[targetImplBSVolatilities.length];
		try {
			optimizer.setValues(bestParameters, finalValues);
		} catch (SolverException e) {}

		System.out.println("\nValuation on calibrated model:");
		double deviationSum			= 0.0;
		double deviationSquaredSum	= 0.0;

		for(int i = 0; i < finalValues.length; i++){
			double valueTarget = targetPrices[i];
			double valueModel = finalValues[i];
			double error = valueModel-valueTarget;
			deviationSum += error;
			deviationSquaredSum += error*error;
			System.out.println("Model: " + formatterValue.format(valueModel) + "\t Target: " + formatterValue.format(valueTarget) + "\t Deviation: " + formatterDeviation.format(error) + "\t" + options[i]);
		}

		double averageDeviation = deviationSum/targetImplBSVolatilities.length;
		System.out.println("Mean Deviation:" + formatterValue.format(averageDeviation));
		System.out.println("RMS Error.....:" + formatterValue.format(Math.sqrt(deviationSquaredSum/targetImplBSVolatilities.length)));
		System.out.println("__________________________________________________________________________________________\n");

		Assert.assertEquals(0.0, Math.abs(averageDeviation), delta);
	}
}
