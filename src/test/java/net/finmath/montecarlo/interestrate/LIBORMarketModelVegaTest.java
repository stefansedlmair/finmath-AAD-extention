package net.finmath.montecarlo.interestrate;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;

import net.finmath.exception.CalculationException;
import net.finmath.marketdata.model.curves.DiscountCurveFromForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.BrownianMotionInterface;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.montecarlo.automaticdifferentiation.backward.RandomVariableDifferentiableAADFactory;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCorrelationModelExponentialDecay;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCovarianceModelFromVolatilityAndCorrelation;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModelFromGivenMatrix;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModelPiecewiseConstant;
import net.finmath.montecarlo.interestrate.products.AbstractLIBORMonteCarloProduct;
import net.finmath.montecarlo.interestrate.products.Swaption;
import net.finmath.montecarlo.process.ProcessEulerScheme;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationInterface;

public class LIBORMarketModelVegaTest {

	private static DecimalFormat formatterValue		= new DecimalFormat(" 000.000%;-##0.000%", new DecimalFormatSymbols(Locale.ENGLISH));
	private static DecimalFormat formatterParam		= new DecimalFormat(" #0.000;-#0.000", new DecimalFormatSymbols(Locale.ENGLISH));
	
	LIBORModelMonteCarloSimulationInterface liborMarketModelMonteCarloSimulation;
	AbstractLIBORMonteCarloProduct product;
	
	
	public LIBORMarketModelVegaTest() throws CalculationException {
		
		AbstractRandomVariableFactory randomVariableFactory = new RandomVariableDifferentiableAADFactory();
		int numberOfPaths				= (int) 1E3;
		int numberOfFactors				= 1;
		double correlationDecayParam	= 4.0;
		
		
		liborMarketModelMonteCarloSimulation =  createLIBORMarketModel(randomVariableFactory, numberOfPaths, numberOfFactors, correlationDecayParam);	
		
		double liborPeriodLength	= 0.5;
		double liborRateTimeHorzion	= 40.0;
		TimeDiscretization liborPeriodDiscretization = new TimeDiscretization(0.0, (int) (liborRateTimeHorzion / liborPeriodLength), liborPeriodLength);

		// Create the forward curve (initial value of the LIBOR market model)
		ForwardCurve forwardCurve = ForwardCurve.createForwardCurveFromForwards(
				"forwardCurve"								/* name of the curve */,
				new double[] {0.5 , 1.0 , 2.0 , 5.0 , 40.0}	/* fixings of the forward */,
				new double[] {0.05, 0.05, 0.05, 0.05, 0.05}	/* forwards */,
				liborPeriodLength							/* tenor / period length */
				);

		int maturityIndex = 60;
		double exerciseDate = liborPeriodDiscretization.getTime(maturityIndex);

		int numberOfPeriods = 20;

		// Create a swaption

		double[] fixingDates = new double[numberOfPeriods];
		double[] paymentDates = new double[numberOfPeriods];
		double[] swapTenor = new double[numberOfPeriods + 1];
		double swapPeriodLength = 0.5;
		String tenorCode = "6M";

		for (int periodStartIndex = 0; periodStartIndex < numberOfPeriods; periodStartIndex++) {
			fixingDates[periodStartIndex] = exerciseDate + periodStartIndex * swapPeriodLength;
			paymentDates[periodStartIndex] = exerciseDate + (periodStartIndex + 1) * swapPeriodLength;
			swapTenor[periodStartIndex] = exerciseDate + periodStartIndex * swapPeriodLength;
		}
		swapTenor[numberOfPeriods] = exerciseDate + numberOfPeriods * swapPeriodLength;

		TimeDiscretizationInterface swapTenorDiscretization = new TimeDiscretization(swapTenor);
		
		// Swaptions swap rate
		double swaprate = net.finmath.marketdata.products.Swap.getForwardSwapRate(swapTenorDiscretization, swapTenorDiscretization, forwardCurve, new DiscountCurveFromForwardCurve(forwardCurve));

		// Set swap rates for each period
		double[] swaprates = new double[numberOfPeriods];
		Arrays.fill(swaprates, swaprate);

		double[] periodLengths = new double[numberOfPeriods];
		Arrays.fill(periodLengths, swapPeriodLength);

		double[] periodNotionals = new double[numberOfPeriods];
		Arrays.fill(periodNotionals, 1.0);

		boolean[] isPeriodStartDateExerciseDate = new boolean[numberOfPeriods];
		Arrays.fill(isPeriodStartDateExerciseDate, true);
		
		product = new Swaption(exerciseDate, swapTenorDiscretization, swaprate);
	}

	public LIBORModelMonteCarloSimulationInterface createLIBORMarketModel(
			AbstractRandomVariableFactory randomVariableFactory,
			int numberOfPaths, int numberOfFactors, double correlationDecayParam) throws CalculationException {

		/*
		 * Create the libor tenor structure and the initial values
		 */
		double liborPeriodLength	= 0.5;
		double liborRateTimeHorzion	= 40.0;
		TimeDiscretization liborPeriodDiscretization = new TimeDiscretization(0.0, (int) (liborRateTimeHorzion / liborPeriodLength), liborPeriodLength);

		// Create the forward curve (initial value of the LIBOR market model)
		ForwardCurve forwardCurve = ForwardCurve.createForwardCurveFromForwards(
				"forwardCurve"								/* name of the curve */,
				new double[] {0.5 , 1.0 , 2.0 , 5.0 , 40.0}	/* fixings of the forward */,
				new double[] {0.05, 0.05, 0.05, 0.05, 0.05}	/* forwards */,
				liborPeriodLength							/* tenor / period length */
				);

		/*
		 * Create a simulation time discretization
		 */
		double lastTime	= 40.0;
		double dt		= 0.125;

		TimeDiscretization timeDiscretization = new TimeDiscretization(0.0, (int) (lastTime / dt), dt);

		/* volatility model from piecewise constant interpolated matrix */
		TimeDiscretizationInterface volatilitySurfaceDiscretization = new TimeDiscretization(0.00, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0); 
		double[] initialVolatility = new double[] { 0.50 / 100 };
		LIBORVolatilityModel volatilityModel = new LIBORVolatilityModelPiecewiseConstant(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, volatilitySurfaceDiscretization, volatilitySurfaceDiscretization, initialVolatility, true);
		
//		/* volatility model from given matrix */
//		double initialVolatility = 0.005;
//		double[][] volatility = new double[timeDiscretization.getNumberOfTimeSteps()][liborPeriodDiscretization.getNumberOfTimeSteps()];
//		for(int i = 0; i < timeDiscretization.getNumberOfTimeSteps(); i++) Arrays.fill(volatility[i], initialVolatility);
//		LIBORVolatilityModel volatilityModel = new LIBORVolatilityModelFromGivenMatrix(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, volatility);

		/*
		 * Create a correlation model rho_{i,j} = exp(-a * abs(T_i-T_j))
		 */
		LIBORCorrelationModelExponentialDecay correlationModel = new LIBORCorrelationModelExponentialDecay(
				timeDiscretization, liborPeriodDiscretization, numberOfFactors,
				correlationDecayParam);


		/*
		 * Combine volatility model and correlation model to a covariance model
		 */
		LIBORCovarianceModelFromVolatilityAndCorrelation covarianceModel =
				new LIBORCovarianceModelFromVolatilityAndCorrelation(timeDiscretization,
						liborPeriodDiscretization, volatilityModel, correlationModel);

		// BlendedLocalVolatlityModel (future extension)
		//		AbstractLIBORCovarianceModel covarianceModel2 = new BlendedLocalVolatlityModel(covarianceModel, 0.00, false);

		// Set model properties
		Map<String, String> properties = new HashMap<String, String>();

		// Choose the simulation measure
		properties.put("measure", LIBORMarketModel.Measure.SPOT.name());

		// Choose log normal model
		properties.put("stateSpace", LIBORMarketModel.StateSpace.NORMAL.name());

		// Empty array of calibration items - hence, model will use given covariance
		LIBORMarketModel.CalibrationItem[] calibrationItems = new LIBORMarketModel.CalibrationItem[0];

		/*
		 * Create corresponding LIBOR Market Model
		 */
		LIBORMarketModelInterface liborMarketModel = new LIBORMarketModel(liborPeriodDiscretization, null, forwardCurve, new DiscountCurveFromForwardCurve(forwardCurve), randomVariableFactory, covarianceModel, calibrationItems, properties);

		BrownianMotionInterface brownianMotion = new net.finmath.montecarlo.BrownianMotion(timeDiscretization, numberOfFactors, numberOfPaths, 3141 /* seed */);

		ProcessEulerScheme process = new ProcessEulerScheme(brownianMotion, ProcessEulerScheme.Scheme.EULER);

		return new LIBORModelMonteCarloSimulation(liborMarketModel, process);
	}
	
	@Test
	public void covarianceParameterDerivativeTest() throws CalculationException {
			
		final double evaluationTime = 0.0;
		final RandomVariableInterface productValues = product.getValue(evaluationTime, liborMarketModelMonteCarloSimulation);
		
		final AbstractLIBORCovarianceModelParametric covarianceModel = (AbstractLIBORCovarianceModelParametric) ((LIBORMarketModelInterface)liborMarketModelMonteCarloSimulation.getModel()).getCovarianceModel();
		final double[] parameter = covarianceModel.getParameter();
		final long[] parameterID = covarianceModel.getParameterID();
		
		//AAD:
		long startAAD = System.currentTimeMillis();
		Map<Long, RandomVariableInterface> gradientAAD = ((RandomVariableDifferentiableInterface) productValues).getGradient();
		long endAAD = System.currentTimeMillis();

		//FD:
		long startFD = System.currentTimeMillis();
		
		double finiteDifferencesStepSize = 1E-4;
		Map<Long, RandomVariableInterface> gradientFD = new HashMap<>();
		for(int parameterIndex = 0; parameterIndex < parameterID.length; parameterIndex++) {
			double[] bumpedParameter = parameter.clone();
			bumpedParameter[parameterIndex] += finiteDifferencesStepSize;
			
			Map<String, Object> dataModified = new HashMap<>();
			dataModified.put("covarianceModel", covarianceModel.getCloneWithModifiedParameters(bumpedParameter));
			RandomVariableInterface productValuesPlus = product.getValue(evaluationTime, liborMarketModelMonteCarloSimulation.getCloneWithModifiedData(dataModified));
			RandomVariableInterface partialDerivative = productValuesPlus.sub(productValues).div(finiteDifferencesStepSize);
			
			gradientFD.put(parameterID[parameterIndex], partialDerivative);
		}
		long endFD = System.currentTimeMillis();

		// comparison
		RandomVariableInterface zero = liborMarketModelMonteCarloSimulation.getRandomVariableForConstant(0.0);
		
		System.out.println("ParameterID      AAD Value               FD Value                Deviation");
		Map<Long, RandomVariableInterface> gradientDifferences = new HashMap<>();
		for(long key : parameterID) {	
//			try {
				RandomVariableInterface aad = gradientAAD.getOrDefault(key, zero);
				RandomVariableInterface fd = gradientFD.getOrDefault(key, zero);				
				RandomVariableInterface diff = aad.sub(fd);				

				gradientDifferences.put(key, diff);
				
				System.out.println(key + "\t\t" + formatterValue.format(aad.getAverage()) + 
										 "\t\t" + formatterValue.format(fd.getAverage()) + 
										 "\t\t" +  formatterValue.format(diff.getAverage()));
//			} catch (Exception e) {
////				if(!gradientAAD.containsKey(key)) System.out.println("gradientAAD does not contain partial derivative wrt " + key + "!");
////				if(!gradientFD.containsKey(key)) System.out.println("gradientFD does not contain partial derivative wrt " + key + "!");
//			}			
		}
		
		System.out.println();
		System.out.println("Calculation Time: \n" +
							"AAD...:" + formatterParam.format((endAAD - startAAD)/(1E3)) + "\n" + 
							"FD....:" + formatterParam.format((endFD - startFD)/(1E3)));
	
		
		double errorRMS = 0.0;
		for(long key : parameterID) {
			double error = gradientDifferences.getOrDefault(key, zero).getAverage();
			errorRMS += error * error;
		}
		errorRMS = Math.sqrt(errorRMS / (double) parameterID.length);
		
		System.out.println("Root-Mean-Square Error: " + formatterValue.format(errorRMS));
		Assert.assertEquals(0.0, errorRMS, 1E-2);
	}
	
}
