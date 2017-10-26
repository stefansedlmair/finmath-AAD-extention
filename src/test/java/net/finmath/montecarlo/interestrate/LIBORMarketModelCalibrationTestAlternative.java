/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 16.01.2015
 */
package net.finmath.montecarlo.interestrate;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.time.LocalDate;
import java.time.Month;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import net.finmath.exception.CalculationException;
import net.finmath.marketdata.model.curves.DiscountCurveFromForwardCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterface;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurveInterface;
import net.finmath.marketdata.products.Swap;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.BrownianMotionInterface;
import net.finmath.montecarlo.IndependentIncrementsInterface;
import net.finmath.montecarlo.automaticdifferentiation.backward.RandomVariableDifferentiableAADFactory;
import net.finmath.montecarlo.interestrate.LIBORMarketModel.CalibrationItem;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric.OptimizerDerivativeType;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric.OptimizerSolverType;
import net.finmath.montecarlo.interestrate.modelplugins.DisplacedLocalVolatilityModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCorrelationModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCorrelationModelExponentialDecay;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCovarianceModelFromVolatilityAndCorrelation;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModelFourParameterExponentialForm;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModelPiecewiseConstant;
import net.finmath.montecarlo.interestrate.products.AbstractLIBORMonteCarloProduct;
import net.finmath.montecarlo.interestrate.products.Swaption;
import net.finmath.montecarlo.process.ProcessEulerScheme;
import net.finmath.montecarlo.process.ProcessEulerScheme.Scheme;
import net.finmath.optimizer.OptimizerFactory;
import net.finmath.optimizer.OptimizerFactory.OptimizerType;
import net.finmath.optimizer.OptimizerFactoryInterface;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationInterface;
import net.finmath.time.businessdaycalendar.BusinessdayCalendarExcludingTARGETHolidays;
import net.finmath.time.daycount.DayCountConvention_ACT_365;

/**
 * This class tests the LIBOR market model and products.
 * 
 * @author Christian Fries
 * @author Stefan Sedlmair
 */

@RunWith(Parameterized.class)
public class LIBORMarketModelCalibrationTestAlternative {

	private static DecimalFormat formatterValue		= new DecimalFormat(" ##0.000%;-##0.000%", new DecimalFormatSymbols(Locale.ENGLISH));
	private static DecimalFormat formatterParam		= new DecimalFormat(" #0.000;-#0.000", new DecimalFormatSymbols(Locale.ENGLISH));
	private static DecimalFormat formatterDeviation	= new DecimalFormat(" 0.00000E00;-0.00000E00", new DecimalFormatSymbols(Locale.ENGLISH));

	int seed;
	int numberOfPaths;
	int numberOfFactors;

	TimeDiscretizationInterface timeDiscretization ;
	TimeDiscretizationInterface liborPeriodDiscretization;

	ForwardCurveInterface forwardRateCurve;
	DiscountCurveInterface discountCurve;

	AbstractRandomVariableFactory randomVariableFactory;

	@Parameters(name="{0}-{1}")
	public static Collection<Object[]> data() {

		Collection<Object[]> config = new ArrayList<>();
		
		int maxIterations = 100;
		double errorTolerance = 1E-4;
		
		config.add(new Object[] {OptimizerSolverType.VECTOR, OptimizerDerivativeType.ADJOINT_ALGORITHMIC_DIFFERENCIATION,
				new OptimizerFactory(OptimizerType.LevenbergMarquardt, maxIterations, errorTolerance)});
		config.add(new Object[] {OptimizerSolverType.SKALAR, OptimizerDerivativeType.ADJOINT_ALGORITHMIC_DIFFERENCIATION,
				new OptimizerFactory(OptimizerType.LevenbergMarquardt, maxIterations, errorTolerance)});
//		config.add(new Object[] {OptimizerSolverType.VECTOR, OptimizerDerivativeType.FINITE_DIFFERENCES});
//		config.add(new Object[] {OptimizerSolverType.SKALAR, OptimizerDerivativeType.FINITE_DIFFERENCES});
		
		return config;
	}	

	private final OptimizerSolverType solverType;
	private final OptimizerDerivativeType derivativeType;
	private final OptimizerFactoryInterface optimizerFactory;
	
	public LIBORMarketModelCalibrationTestAlternative(OptimizerSolverType solverType, OptimizerDerivativeType derivativeType, OptimizerFactoryInterface optimizerFactory) {

		this.optimizerFactory = optimizerFactory;
		
		System.out.println(solverType + " - " + derivativeType);
		
		double simulationStart 		= 0.0;
		double simulationEnd 		= 40.0;
		double liborPeriodLength 	= 0.25;

		// time discretizations
		timeDiscretization 			= new TimeDiscretization(simulationStart, (int) (simulationEnd / liborPeriodLength), liborPeriodLength);
		liborPeriodDiscretization 	= timeDiscretization;

		// initial forward curve
		forwardRateCurve = ForwardCurve.createForwardCurveFromForwards(
				"forwardCurve"								/* name of the curve */,
				new double[] {0.5 , 1.0 , 2.0 , 5.0 , 40.0}	/* fixings of the forward */,
				new double[] {0.05, 0.05, 0.05, 0.05, 0.05}	/* forwards */,
				liborPeriodLength							/* tenor / period length */
				);
		discountCurve 			= new DiscountCurveFromForwardCurve(forwardRateCurve);

		// first Parameters
		seed 					= 1234;
		numberOfPaths 			= (int) 1E3;
		numberOfFactors 		= 1;

		// factory
		this.randomVariableFactory = new RandomVariableDifferentiableAADFactory();

		// properties
		this.solverType 		= solverType;
		this.derivativeType 	= derivativeType;
	}

	@Test
	public void atmSwaptionCalibration() throws CalculationException {

		// Simulation Settings
		int seed				= 1234;
		int numberOfPaths		= (int) 1E3;
		int numberOfFactors		= 1;

		// Optimizer Settings
		Double accuracy 		= new Double(1E-4);	// Lower accuracy to reduce runtime of the unit test
		int maxIterations 		= 100;
		int numberOfThreads 	= 2;

		// brownian motion
		final BrownianMotionInterface brownianMotion = new net.finmath.montecarlo.BrownianMotion(timeDiscretization, numberOfFactors, numberOfPaths, seed);

		/* volatility model from piecewise constant interpolated matrix */
		TimeDiscretizationInterface volatilitySurfaceDiscretization = new TimeDiscretization(0.00, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0); 
		double[] initialVolatility = new double[] { 0.50 / 100 };
		LIBORVolatilityModel volatilityModel = new LIBORVolatilityModelPiecewiseConstant(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, volatilitySurfaceDiscretization, volatilitySurfaceDiscretization, initialVolatility, true);

		/* Correlation Model with exponential decay */
		LIBORCorrelationModel correlationModel = new LIBORCorrelationModelExponentialDecay(timeDiscretization, liborPeriodDiscretization, numberOfFactors, 0.05, false);

		// Create a covariance model
		AbstractLIBORCovarianceModelParametric covarianceModelParametric = new LIBORCovarianceModelFromVolatilityAndCorrelation(timeDiscretization, liborPeriodDiscretization, volatilityModel, correlationModel);

		// Create blended local volatility model with fixed parameter (0=lognormal, > 1 = almost a normal model).			
		AbstractLIBORCovarianceModelParametric covarianceModelDisplaced = new DisplacedLocalVolatilityModel(randomVariableFactory, covarianceModelParametric, 1.0/0.25, false /* isCalibrateable */);

		// Set model properties
		Map<String, Object> properties = new HashMap<String, Object>();
		properties.put("measure", LIBORMarketModel.Measure.SPOT.name());
		properties.put("stateSpace", LIBORMarketModel.StateSpace.NORMAL.name());

		Map<String, Object> calibrationParameters = new HashMap<String, Object>();
		calibrationParameters.put("accuracy", accuracy);
		calibrationParameters.put("brownianMotion", brownianMotion);
		calibrationParameters.put("optimizerFactory", optimizerFactory);
		calibrationParameters.put("parameterStep", new Double(1E-4));

		calibrationParameters.put("solverType", solverType);
		calibrationParameters.put("derivativeType", derivativeType);
		calibrationParameters.put("scheme", Scheme.EULER);

		properties.put("calibrationParameters", calibrationParameters);

		Object[] calibrationItemsAndNames = getLMMCalibrationItems();
		CalibrationItem[] calibrationItems = (CalibrationItem[]) calibrationItemsAndNames[0];
		String[] calibrationItemNames = (String[]) calibrationItemsAndNames[1];
		
		long millisCalibrationStart = System.currentTimeMillis();

		// Create corresponding LIBOR Market Model
		LIBORModelInterface liborMarketModelCalibrated = new LIBORMarketModel(
				liborPeriodDiscretization,
				null /*analytic model*/,
				forwardRateCurve,
				discountCurve,
				randomVariableFactory,
				covarianceModelDisplaced,
				calibrationItems,
				properties);

		long millisCalibrationEnd = System.currentTimeMillis();

		// process
		ProcessEulerScheme eulerScheme = new ProcessEulerScheme(brownianMotion);

		// monte carlo simulation 
		LIBORModelMonteCarloSimulation liborModelCalibratedMonteCarloSimulation = new LIBORModelMonteCarloSimulation(liborMarketModelCalibrated, eulerScheme);
		
		System.out.println("\nCalibrated parameters are:");
		AbstractLIBORCovarianceModelParametric calibratedCovarianceModel = (AbstractLIBORCovarianceModelParametric) ((LIBORMarketModel) liborMarketModelCalibrated).getCovarianceModel();

		double[] param = calibratedCovarianceModel.getParameter();
		for (double p : param) System.out.println(formatterParam.format(p));
		System.out.println("\nValuation on calibrated model:");
		int numberOfProducts = calibrationItems.length;
		double deviationSum			= 0.0;
		double deviationSquaredSum	= 0.0;
		for (int i = 0; i < numberOfProducts; i++) {
			CalibrationItem calibrationItem = calibrationItems[i];
			try {
				double valueModel = calibrationItem.calibrationProduct.getValue(liborModelCalibratedMonteCarloSimulation);
				double valueTarget = calibrationItem.calibrationTargetValue;
				double error = valueModel-valueTarget;
				deviationSum += error;
				deviationSquaredSum += error*error;
				System.out.println(calibrationItemNames[i] + "\t" + "Model: " + formatterValue.format(valueModel) + "\t Target: " + formatterValue.format(valueTarget) + "\t Deviation: " + formatterDeviation.format(valueModel-valueTarget));// + "\t" + calibrationProduct.toString());
			}
			catch(Exception e) {
			}
		}

		System.out.println("Calibration of prices... " + (millisCalibrationEnd-millisCalibrationStart)/1E3 + "s");
		System.out.println("Number of Iterations.... " + covarianceModelDisplaced.getCalibrationOptimizer().getIterations());
		
		double averageDeviation = deviationSum/numberOfProducts;
		System.out.println("Mean Deviation:" + formatterValue.format(averageDeviation));
		System.out.println("RMS Error.....:" + formatterValue.format(Math.sqrt(deviationSquaredSum/numberOfProducts)));
		System.out.println("__________________________________________________________________________________________\n");

		Assert.assertEquals(0.0, averageDeviation, 1E-2);

	}

	private Object[] getLMMCalibrationItems() throws CalculationException {

		// parameters	
		double a = 0.1;
		double b = 0.1;
		double c = 0.1;
		double d = 0.1;

		double alpha = 4.0;

		String[] atmExpiries = { "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y" };
		String[] atmTenors = { "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y" };

		// model		
		LIBORVolatilityModel volatilityModel = new LIBORVolatilityModelFourParameterExponentialForm(timeDiscretization, liborPeriodDiscretization, a, b, c, d, false);
		LIBORCorrelationModel correlationModel = new LIBORCorrelationModelExponentialDecay(timeDiscretization, liborPeriodDiscretization, numberOfFactors, alpha);

		LIBORCovarianceModelFromVolatilityAndCorrelation covarianceModel = new LIBORCovarianceModelFromVolatilityAndCorrelation(timeDiscretization, liborPeriodDiscretization, volatilityModel, correlationModel);

		LIBORMarketModel liborMarketModel = new LIBORMarketModel(liborPeriodDiscretization, forwardRateCurve, covarianceModel);

		// process
		IndependentIncrementsInterface brownianMotion = new BrownianMotion(timeDiscretization, numberOfFactors, numberOfPaths, seed);
		ProcessEulerScheme eulerScheme = new ProcessEulerScheme(brownianMotion);

		// monte carlo simulation
		LIBORModelMonteCarloSimulation liborModelMonteCarloSimulation = new LIBORModelMonteCarloSimulation(liborMarketModel, eulerScheme);

		// products
		double	swapPeriodLength	= 0.5;

		ArrayList<CalibrationItem> calibrationItems = new ArrayList<>();
		ArrayList<String> calibrationItemNames = new ArrayList<>();
		
		LocalDate referenceDate = LocalDate.of(2016, Month.SEPTEMBER, 30); 
		BusinessdayCalendarExcludingTARGETHolidays cal = new BusinessdayCalendarExcludingTARGETHolidays();
		DayCountConvention_ACT_365 modelDC = new DayCountConvention_ACT_365();
		for(int i=0; i<atmTenors.length; i++ ) {

			LocalDate exerciseDate = cal.createDateFromDateAndOffsetCode(referenceDate, atmExpiries[i]);
			LocalDate tenorEndDate = cal.createDateFromDateAndOffsetCode(exerciseDate, atmTenors[i]);
			double	exercise		= modelDC.getDaycountFraction(referenceDate, exerciseDate);
			double	tenor			= modelDC.getDaycountFraction(exerciseDate, tenorEndDate);

			// We consider an idealized tenor grid (alternative: adapt the model grid)
			exercise	= Math.round(exercise/0.25)*0.25;
			tenor		= Math.round(tenor/0.25)*0.25;

			if(exercise < 1.0) continue;

			int numberOfPeriods = (int)Math.round(tenor / swapPeriodLength);

			TimeDiscretizationInterface swapTenor = new TimeDiscretization(exercise, numberOfPeriods + 1, swapPeriodLength);

			double swapRate = Swap.getForwardSwapRate(swapTenor, swapTenor, forwardRateCurve); 

			try {
				AbstractLIBORMonteCarloProduct product = new Swaption(exercise, swapTenor, swapRate);
				double targetValue = product.getValue(liborModelMonteCarloSimulation);
				double weight = 1.0;

				calibrationItems.add(new CalibrationItem(product, targetValue, weight));
				calibrationItemNames.add(atmExpiries[i] + "\t" + atmTenors[i]);
			}
			catch (Exception e) {
				// if calculation of product fails do not include for calibration.
			}
		}

		int numberOfProducts = calibrationItems.size();
		return new Object[]{
				calibrationItems.toArray(new CalibrationItem[numberOfProducts]),
				calibrationItemNames.toArray(new String[numberOfProducts])};		
		
	}

}