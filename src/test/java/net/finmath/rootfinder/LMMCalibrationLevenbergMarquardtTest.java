package net.finmath.rootfinder;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;
import java.util.Map;

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;

import net.finmath.exception.CalculationException;
import net.finmath.marketdata.model.curves.DiscountCurveFromForwardCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterface;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurveInterface;
import net.finmath.marketdata.products.Swap;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.BrownianMotionInterface;
import net.finmath.montecarlo.BrownianMotionView;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.montecarlo.automaticdifferentiation.backward.RandomVariableDifferentiableAADFactory;
import net.finmath.montecarlo.automaticdifferentiation.backward.alternative.RandomVariableAADFactory;
import net.finmath.montecarlo.interestrate.LIBORMarketModel;
import net.finmath.montecarlo.interestrate.LIBORMarketModelInterface;
import net.finmath.montecarlo.interestrate.LIBORModelMonteCarloSimulation;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric;
import net.finmath.montecarlo.interestrate.modelplugins.BlendedLocalVolatilityModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCovarianceModelExponentialForm5Param;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCovarianceModelStochasticVolatility;
import net.finmath.montecarlo.interestrate.products.Swaption;
import net.finmath.montecarlo.process.ProcessEulerScheme;
import net.finmath.optimizer.LevenbergMarquardt;
import net.finmath.optimizer.SolverException;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationInterface;

public class LMMCalibrationLevenbergMarquardtTest {


	private static DecimalFormat formatterValue		= new DecimalFormat(" ##0.000%;-##0.000%", new DecimalFormatSymbols(Locale.ENGLISH));
	private static DecimalFormat formatterDeviation	= new DecimalFormat(" 0.00000E00;-0.00000E00", new DecimalFormatSymbols(Locale.ENGLISH));

	ArrayList<Swaption> swaptions;
	ArrayList<Double> targetValues;
	
	ForwardCurve forwardCurve;
	double liborPeriodLength;
	
	
	int numberOfPaths;
	int numberOfFactors;
	
	LevenbergMarquardt optimizer;
	int numberOfIterations;
	int numberOfThreads;
	
	double delta = 1E-2;
	
	public LMMCalibrationLevenbergMarquardtTest() {

		System.out.println("--------------------------------------------------------------------------------------------------");

		
		numberOfPaths		= (int) 1E2;
		numberOfFactors		= 		5;
		
		numberOfIterations 	= 		100;
		numberOfThreads		= 		2;
		
		// calculation of various target Swap-prices for all kinds of different data 

		System.out.println("Fixing Forward Curve...");
		
		double[] fixingTimes = new double[] {
				0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.5, 50.0
		};

		double[] forwardRates = new double[] {
				0.61/100.0,	0.61/100.0,	0.67/100.0,	0.73/100.0,	0.80/100.0,	0.92/100.0,	1.11/100.0,	1.36/100.0,	1.60/100.0,	1.82/100.0,	2.02/100.0,	2.17/100.0,	2.27/100.0,	2.36/100.0,	2.46/100.0,	2.52/100.0,	2.54/100.0,	2.57/100.0,	2.68/100.0,	2.82/100.0,	2.92/100.0,	2.98/100.0,	3.00/100.0,	2.99/100.0,	2.95/100.0,	2.89/100.0,	2.82/100.0,	2.74/100.0,	2.66/100.0,	2.59/100.0,	2.52/100.0,	2.47/100.0,	2.42/100.0,	2.38/100.0,	2.35/100.0,	2.33/100.0,	2.31/100.0,	2.30/100.0,	2.29/100.0,	2.28/100.0,	2.27/100.0,	2.27/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.27/100.0,	2.28/100.0,	2.28/100.0,	2.30/100.0,	2.31/100.0,	2.32/100.0,	2.34/100.0,	2.35/100.0,	2.37/100.0,	2.39/100.0,	2.42/100.0,	2.44/100.0,	2.47/100.0,	2.50/100.0,	2.52/100.0,	2.56/100.0,	2.59/100.0,	2.62/100.0,	2.65/100.0,	2.68/100.0,	2.72/100.0,	2.75/100.0,	2.78/100.0,	2.81/100.0,	2.83/100.0,	2.86/100.0,	2.88/100.0,	2.91/100.0,	2.93/100.0,	2.94/100.0,	2.96/100.0,	2.97/100.0,	2.97/100.0,	2.97/100.0,	2.97/100.0,	2.97/100.0,	2.96/100.0,	2.95/100.0,	2.94/100.0,	2.93/100.0,	2.91/100.0,	2.89/100.0,	2.87/100.0,	2.85/100.0,	2.83/100.0,	2.80/100.0,	2.78/100.0,	2.75/100.0,	2.72/100.0,	2.69/100.0,	2.67/100.0,	2.64/100.0,	2.64/100.0
		};

		liborPeriodLength = 0.5;

		// Create the forward curve (initial value of the LIBOR market model)
		forwardCurve = ForwardCurve.createForwardCurveFromForwards(
				"forwardCurve"		/* name of the curve */,
				fixingTimes			/* fixings of the forward */,
				forwardRates		/* forwards */,
				liborPeriodLength	/* tenor / period length */
				);


		DiscountCurveInterface discountCurve = new DiscountCurveFromForwardCurve(forwardCurve, liborPeriodLength);



		//---------------------- associate swaption with target volatilities ------------------------------------ 

		System.out.println("Collection Swaptions with their associated target value for calibration...");

		swaptions = new ArrayList<>();
		targetValues = new ArrayList<>();

		double	swapPeriodLength	= 0.5;
		int		numberOfPeriods		= 20;

		double[] smileMoneynesses	= { -0.02,	-0.01, -0.005, -0.0025,	0.0,	0.0025,	0.0050,	0.01,	0.02 };
		double[] smileVolatilities	= { 0.559,	0.377,	0.335,	 0.320,	0.308, 0.298, 0.290, 0.280, 0.270 };

		for(int i=0; i<smileMoneynesses.length; i++ ) {

			double	exerciseDate		= 5.0;
			double	moneyness			= smileMoneynesses[i];
			double	targetVolatility	= smileVolatilities[i]; // TODO what is meant by this volatility? implied Black Volatility?

			TimeDiscretizationInterface swapTenor = new TimeDiscretization(exerciseDate, numberOfPeriods, swapPeriodLength);

			double swapRate = Swap.getForwardSwapRate(swapTenor, swapTenor, forwardCurve, discountCurve);

			Swaption swaption = new Swaption(exerciseDate, swapTenor, swapRate + moneyness);

			swaptions.add(swaption);
			targetValues.add(targetVolatility);
		}

		double[] atmOptionMaturities	= { 2.00, 3.00, 4.00, 5.00, 7.00, 10.00, 15.00, 20.00, 25.00, 30.00 };
		double[] atmOptionVolatilities	= { 0.385, 0.351, 0.325, 0.308, 0.288, 0.279, 0.290, 0.272, 0.235, 0.192 };

		for(int i=0; i<atmOptionMaturities.length; i++ ) {

			double	exerciseDate		= atmOptionMaturities[i];
			double	targetVolatility	= atmOptionVolatilities[i]; // TODO what is meant by this volatility? implied Black Volatility?

			TimeDiscretizationInterface swapTenor = new TimeDiscretization(exerciseDate, numberOfPeriods, swapPeriodLength);

			double swapRate = Swap.getForwardSwapRate(swapTenor, swapTenor, forwardCurve, discountCurve);

			Swaption swaption = new Swaption(exerciseDate, swapTenor, swapRate);	

			swaptions.add(swaption);
			targetValues.add(targetVolatility);
		}
		
		System.out.println();

	}


	@Test
	public void UsingStandardLevenbergMarquardtOptimizer() throws CalculationException{

		AbstractRandomVariableFactory randomVariableFactory = new RandomVariableFactory();
		
		/* 
		 * create a LIBOR Market Model
		 * */
		
		System.out.println("Create Libor Market Model");

		/*
		 * Create the libor tenor structure and the initial values
		 */
		double liborRateTimeHorzion	= 40.0;
		TimeDiscretization liborPeriodDiscretization = new TimeDiscretization(0.0, (int) (liborRateTimeHorzion / liborPeriodLength), liborPeriodLength);

		/*
		 * Create a simulation time discretization
		 */
		double lastTime	= 40.0;
		double dt		= 0.5;
		TimeDiscretization timeDiscretization = new TimeDiscretization(0.0, (int) (lastTime / dt), dt);

		/*
		 * Create Brownian motions 
		 */
		BrownianMotionInterface brownianMotion = new net.finmath.montecarlo.BrownianMotion(timeDiscretization, numberOfFactors + 1, numberOfPaths, 31415 /* seed */);
		BrownianMotionInterface brownianMotionView1 = new BrownianMotionView(brownianMotion, new Integer[] { 0, 1, 2, 3, 4 });
		BrownianMotionInterface brownianMotionView2 = new BrownianMotionView(brownianMotion, new Integer[] { 0, 5 });

		// define the covariance Parameters
		double[] initialCovarianceParametersDoubles = new double[] { 0.20, 0.05, 0.10, 0.05, 0.10};

		double[] targetValuesDouble = new double[targetValues.size()];
		for(int i = 0; i < targetValues.size(); i++)
			targetValuesDouble[i] = targetValues.get(i);

		System.out.println("Start standard Levenberg Marquardt Optimizer:");
		
		optimizer = new LevenbergMarquardt(initialCovarianceParametersDoubles, targetValuesDouble, numberOfIterations, numberOfThreads) {

			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
				RandomVariableInterface[] initialCovarianceParameters = new RandomVariableInterface[parameters.length];
				for(int i = 0; i < initialCovarianceParameters.length; i++)
					initialCovarianceParameters[i] = randomVariableFactory.createRandomVariable(parameters[i]);

				try {
					ArrayList<RandomVariableInterface> swaptionValues = calculateSwaptionPrice(swaptions, timeDiscretization, liborPeriodDiscretization, numberOfFactors, initialCovarianceParameters, brownianMotionView1, brownianMotionView2, forwardCurve);
					for(int i = 0; i < swaptionValues.size(); i++)
						values[i] = swaptionValues.get(i).getAverage();

				} catch (CalculationException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		};

		try {
			optimizer.run();
		} catch (SolverException e) {}
	}

	@Test
	public void UsingTweekedLevenbergMarquardtOptimizer() throws CalculationException{

		AbstractRandomVariableFactory randomVariableFactory = new RandomVariableDifferentiableAADFactory();//new RandomVariableAADFactory(new RandomVariableFactory());
			
		/* 
		 * create a LIBOR Market Model
		 * */

		System.out.println("Create Libor Market Model");

		/*
		 * Create the libor tenor structure and the initial values
		 */
		double liborRateTimeHorzion	= 40.0;
		TimeDiscretization liborPeriodDiscretization = new TimeDiscretization(0.0, (int) (liborRateTimeHorzion / liborPeriodLength), liborPeriodLength);

		/*
		 * Create a simulation time discretization
		 */
		double lastTime	= 40.0;
		double dt		= 0.5;
		TimeDiscretization timeDiscretization = new TimeDiscretization(0.0, (int) (lastTime / dt), dt);

		/*
		 * Create Brownian motions 
		 */
		BrownianMotionInterface brownianMotion = new net.finmath.montecarlo.BrownianMotion(timeDiscretization, numberOfFactors + 1, numberOfPaths, 31415 /* seed */);
		BrownianMotionInterface brownianMotionView1 = new BrownianMotionView(brownianMotion, new Integer[] { 0, 1, 2, 3, 4 });
		BrownianMotionInterface brownianMotionView2 = new BrownianMotionView(brownianMotion, new Integer[] { 0, 5 });

		// define the covariance Parameters
		double[] initialCovarianceParametersDoubles = new double[] { 0.20, 0.05, 0.10, 0.05, 0.10};

		double[] targetValuesDouble = new double[targetValues.size()];
		for(int i = 0; i < targetValues.size(); i++)
			targetValuesDouble[i] = targetValues.get(i);

		System.out.println("Start tweeked Levenberg Marquardt Optimizer");
		
		optimizer = new LevenbergMarquardt(initialCovarianceParametersDoubles, targetValuesDouble, numberOfIterations, numberOfThreads) {

			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
//				System.out.println("setValuesStart");

				RandomVariableInterface[] initialCovarianceParameters = new RandomVariableInterface[parameters.length];
				for(int i = 0; i < initialCovarianceParameters.length; i++)
					initialCovarianceParameters[i] = randomVariableFactory.createRandomVariable(parameters[i]);
				try {
					ArrayList<RandomVariableInterface> swaptionValues = calculateSwaptionPrice(swaptions, timeDiscretization, liborPeriodDiscretization, numberOfFactors, initialCovarianceParameters, brownianMotionView1, brownianMotionView2, forwardCurve);
					for(int swaptionIndex = 0; swaptionIndex < swaptionValues.size(); swaptionIndex++)
						values[swaptionIndex] = swaptionValues.get(swaptionIndex).getAverage();
				} catch (CalculationException e) {
					throw new UnknownError("Error while calculting swaptionValues");
				}
//				System.out.println("setValuesEnd");
			}
			
			@Override
			public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
//				System.out.println("setDerivativesStart");

				RandomVariableDifferentiableInterface[] initialCovarianceParameters = new RandomVariableDifferentiableInterface[parameters.length];
				for(int i = 0; i < initialCovarianceParameters.length; i++)
					initialCovarianceParameters[i] = (RandomVariableDifferentiableInterface) randomVariableFactory.createRandomVariable(parameters[i]);

				try {
					ArrayList<RandomVariableInterface> swaptionValues = calculateSwaptionPrice(swaptions, timeDiscretization, liborPeriodDiscretization, numberOfFactors, initialCovarianceParameters, brownianMotionView1, brownianMotionView2, forwardCurve);
					for(int swaptionIndex = 0; swaptionIndex < swaptionValues.size(); swaptionIndex++){
//						System.out.println("gradientStart");
						Map<Long, RandomVariableInterface> gradient = ((RandomVariableDifferentiableInterface) swaptionValues.get(swaptionIndex)).getGradient();
//						System.out.println("gradientEnd");
						for(int parameterIndex = 0; parameterIndex < initialCovarianceParameters.length; parameterIndex++){
							long id = initialCovarianceParameters[parameterIndex].getID();
							RandomVariableInterface partialDerivative = gradient.get(id);

							derivatives[swaptionIndex][parameterIndex] = partialDerivative.getAverage();
						}
					}			

				} catch (CalculationException e) {}
				
//				System.out.println("setDerivativesEnd");
			}
		};

		try {
			optimizer.run();
		} catch (SolverException e) {}		
	}
	
	private ArrayList<RandomVariableInterface> calculateSwaptionPrice(
			ArrayList<Swaption> swaptions,
			TimeDiscretizationInterface timeDiscretization, 
			TimeDiscretizationInterface liborPeriodDiscretization, 
			int numberOfFactors, 
			RandomVariableInterface[] initialCovarianceParameters, 
			BrownianMotionInterface brownianMotionView1,
			BrownianMotionInterface brownianMotionView2, 
			ForwardCurveInterface forwardRateCurve) throws CalculationException{
		// Create a covariance model
		AbstractLIBORCovarianceModelParametric covarianceModelParametric = new LIBORCovarianceModelExponentialForm5Param(timeDiscretization, liborPeriodDiscretization, numberOfFactors, initialCovarianceParameters );
		// Create blended local volatility model with fixed parameter 0.0 (that is "lognormal").
		AbstractLIBORCovarianceModelParametric covarianceModelBlended = new BlendedLocalVolatilityModel(covarianceModelParametric, 0.0, false);
		// Create stochastic scaling (pass brownianMotionView2 to it)
		AbstractLIBORCovarianceModelParametric covarianceModelStochasticParametric = new LIBORCovarianceModelStochasticVolatility(covarianceModelBlended, brownianMotionView2, 0.01, -0.30, true);
		// create Libor Market Model
		LIBORMarketModelInterface model = new LIBORMarketModel(liborPeriodDiscretization, forwardRateCurve, covarianceModelStochasticParametric);
		// create euler scheme
		ProcessEulerScheme process = new ProcessEulerScheme(brownianMotionView1);
		// create ModelMonteCarloSimulation
		LIBORModelMonteCarloSimulation liborMarketModelMonteCarloSimulation =  new LIBORModelMonteCarloSimulation(model, process);

		// evaluate swaptions
		ArrayList<RandomVariableInterface> swaptionValues = new ArrayList<>();
		for(int i = 0; i < swaptions.size(); i++){
			Swaption swaption = swaptions.get(i);
			RandomVariableInterface swaptionValue = swaption.getValue(0.0, liborMarketModelMonteCarloSimulation);
			swaptionValues.add(swaptionValue);
		}

		return swaptionValues;
	}
	
	@After
	public void printResults(){
		double[] bestParameters = optimizer.getBestFitParameters();
		double[] finalValues = new double[targetValues.size()];
		try {
			optimizer.setValues(bestParameters, finalValues);
		} catch (SolverException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		System.out.println(Arrays.toString(bestParameters));
		System.out.println("\nValuation on calibrated model:");
		double deviationSum			= 0.0;
		double deviationSquaredSum	= 0.0;

		for(int i = 0; i < finalValues.length; i++){
			double valueTarget = targetValues.get(i);
			double valueModel = finalValues[i];
			double error = valueModel-valueTarget;
			deviationSum += error;
			deviationSquaredSum += error*error;
			System.out.println("Model: " + formatterValue.format(valueModel) + "\t Target: " + formatterValue.format(valueTarget) + "\t Deviation: " + formatterDeviation.format(error));
		}

		double averageDeviation = deviationSum/targetValues.size();
		System.out.println("Mean Deviation:" + formatterValue.format(averageDeviation));
		System.out.println("RMS Error.....:" + formatterValue.format(Math.sqrt(deviationSquaredSum/targetValues.size())));
		System.out.println("__________________________________________________________________________________________\n");

		Assert.assertEquals(0.0, Math.abs(averageDeviation), delta);
	}

}
