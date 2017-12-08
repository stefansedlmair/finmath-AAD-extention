/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 16.01.2015
 */
package net.finmath.montecarlo.interestrate;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.Month;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import net.finmath.exception.CalculationException;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.functions.FileManagement;
import net.finmath.marketdata.calibration.ParameterObjectInterface;
import net.finmath.marketdata.calibration.Solver;
import net.finmath.marketdata.model.AnalyticModel;
import net.finmath.marketdata.model.AnalyticModelInterface;
import net.finmath.marketdata.model.curves.Curve.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.Curve.InterpolationEntity;
import net.finmath.marketdata.model.curves.Curve.InterpolationMethod;
import net.finmath.marketdata.model.curves.CurveInterface;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveFromForwardCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterface;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurveFromDiscountCurve;
import net.finmath.marketdata.model.curves.ForwardCurveInterface;
import net.finmath.marketdata.products.AnalyticProductInterface;
import net.finmath.marketdata.products.Swap;
import net.finmath.marketdata.products.SwapAnnuity;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.BrownianMotionInterface;
import net.finmath.montecarlo.BrownianMotionView;
import net.finmath.montecarlo.RandomVariable;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableFunctionalFactory;
import net.finmath.montecarlo.interestrate.LIBORMarketModel.CalibrationItem;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric.OptimizerDerivativeType;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric.OptimizerSolverType;
import net.finmath.montecarlo.interestrate.modelplugins.DisplacedLocalVolatilityModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCorrelationModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCorrelationModelExponentialDecay;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCovarianceModelFromVolatilityAndCorrelation;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCovarianceModelStochasticVolatility;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModelPiecewiseConstant;
import net.finmath.montecarlo.interestrate.products.ATMSwaption;
import net.finmath.montecarlo.interestrate.products.AbstractLIBORMonteCarloProduct;
import net.finmath.montecarlo.interestrate.products.SwaptionSimple;
import net.finmath.montecarlo.interestrate.products.SwaptionSimple.ValueUnit;
import net.finmath.montecarlo.process.ProcessEulerScheme;
import net.finmath.montecarlo.process.ProcessEulerScheme.Scheme;
import net.finmath.optimizer.OptimizerFactory;
import net.finmath.optimizer.OptimizerFactory.OptimizerType;
import net.finmath.optimizer.OptimizerFactoryInterface;
import net.finmath.optimizer.OptimizerInterfaceAAD;
import net.finmath.optimizer.SolverException;
import net.finmath.time.ScheduleGenerator;
import net.finmath.time.ScheduleInterface;
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
public class LIBORMarketModelCalibrationTest {

	private static DecimalFormat formatterValue		= new DecimalFormat(" ##0.000%;-##0.000%", new DecimalFormatSymbols(Locale.ENGLISH));
	private static DecimalFormat formatterParam		= new DecimalFormat(" #0.00000;-#0.00000", new DecimalFormatSymbols(Locale.ENGLISH));
	private static DecimalFormat formatterDeviation	= new DecimalFormat(" 0.00000E00;-0.00000E00", new DecimalFormatSymbols(Locale.ENGLISH));

	@Parameters(name="{3}-{2}-{1}")
	public static Collection<Object[]> data() {

		Collection<Object[]> config = new ArrayList<>();

		// Caibration for VOLATILITYNORMALS
		// vector valued calibration
		config.add(new Object[] {OptimizerSolverType.Vector, OptimizerDerivativeType.FiniteDifferences, OptimizerType.LevenbergMarquardt, ValueUnit.VOLATILITYNORMAL});
		config.add(new Object[] {OptimizerSolverType.Vector, OptimizerDerivativeType.AlgorithmicDifferentiation, OptimizerType.LevenbergMarquardt, ValueUnit.VOLATILITYNORMAL});
		config.add(new Object[] {OptimizerSolverType.Vector, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.LevenbergMarquardt, ValueUnit.VOLATILITYNORMAL});

		// scalar valued calibration
		config.add(new Object[] {OptimizerSolverType.Scalar, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.Levenberg, ValueUnit.VOLATILITYNORMAL});
		config.add(new Object[] {OptimizerSolverType.Scalar, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.SimpleGradientDescent, ValueUnit.VOLATILITYNORMAL});
		config.add(new Object[] {OptimizerSolverType.Scalar, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.GradientDescentArmijo, ValueUnit.VOLATILITYNORMAL});
		config.add(new Object[] {OptimizerSolverType.Scalar, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.TruncatedGaussNetwon, ValueUnit.VOLATILITYNORMAL});
		config.add(new Object[] {OptimizerSolverType.Scalar, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.BroydenFletcherGoldfarbShanno, ValueUnit.VOLATILITYNORMAL});

//		// Caibration for VALUES
//		// vector valued calibration
		config.add(new Object[] {OptimizerSolverType.Vector, OptimizerDerivativeType.FiniteDifferences, OptimizerType.LevenbergMarquardt, ValueUnit.VALUE});
		config.add(new Object[] {OptimizerSolverType.Vector, OptimizerDerivativeType.AlgorithmicDifferentiation, OptimizerType.LevenbergMarquardt, ValueUnit.VALUE});
		config.add(new Object[] {OptimizerSolverType.Vector, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.LevenbergMarquardt, ValueUnit.VALUE});
//
//		// scalar valued calibration
		config.add(new Object[] {OptimizerSolverType.Scalar, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.Levenberg, ValueUnit.VALUE});
		config.add(new Object[] {OptimizerSolverType.Scalar, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.SimpleGradientDescent, ValueUnit.VALUE});
		config.add(new Object[] {OptimizerSolverType.Scalar, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.GradientDescentArmijo, ValueUnit.VALUE});
		config.add(new Object[] {OptimizerSolverType.Scalar, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.TruncatedGaussNetwon, ValueUnit.VALUE});
		config.add(new Object[] {OptimizerSolverType.Scalar, OptimizerDerivativeType.AdjointAlgorithmicDifferentiation, OptimizerType.BroydenFletcherGoldfarbShanno, ValueUnit.VALUE});

		return config;
	}	

	private Map<String, Object> testProperties;
	
	public LIBORMarketModelCalibrationTest(OptimizerSolverType solverType, OptimizerDerivativeType derivativeType, OptimizerType optimizerType, ValueUnit valueUnit) {

		System.out.println("\n" + solverType + " - " + optimizerType + " - " + derivativeType + "\n");

		Map<String, Object> factoryProperties = new HashMap<>();
		
		switch(derivativeType) {
		case AdjointAlgorithmicDifferentiation:
			factoryProperties.put("enableAAD", 	true);
			factoryProperties.put("enableAD", 	false);
			break;
		case AlgorithmicDifferentiation:
			factoryProperties.put("enableAAD", 	false);
			factoryProperties.put("enableAD",	true);
			break;
		default:
			factoryProperties.put("enableAAD", 	false);
			factoryProperties.put("enableAD", 	false);
		}

		AbstractRandomVariableFactory randomVariableFactory = new RandomVariableDifferentiableFunctionalFactory(new RandomVariableFactory(), factoryProperties);
				
		testProperties = new HashMap<>();

		testProperties.put("RandomVariableFactory", randomVariableFactory);
		testProperties.put("OptimizerType", 		optimizerType);
		testProperties.put("DerivativeType", 		derivativeType);
		testProperties.put("SolverType", 			solverType);
		testProperties.put("ValueUnit", 			valueUnit);

		testProperties.put("numberOfPathsATM", 			(int) 1E3);
		testProperties.put("numberOfPathsSwaptionSmile",(int) 1E3);
	
		
		testProperties.put("numberOfThreads", 	4); /*max Threads CIP90/91: 16/12 */
		testProperties.put("maxIterations", 	400);
		testProperties.put("errorTolerance", 	1E-4);

//		testProperties.put("maxRunTime", 		(long) (30 * /*min->sec*/ 60 * /*sec->millis*/ 1E3));	
		
		testProperties.put("stepSize", 	0.0001);

		
	}

	@Test
	public void testSwaptionSmileCalibration() throws CalculationException, SolverException {
		SwaptionSmileCalibration(testProperties);
	}
	
	
	/**
	 * Brute force Monte-Carlo calibration of swaptions.
	 * 
	 * @throws CalculationException
	 * @throws SolverException
	 */
	@Test
	public void testATMSwaptionCalibration() throws CalculationException, SolverException {
		ATMSwaptionCalibration(testProperties);
	}
	
	public static void SwaptionSmileCalibration(Map<String, Object> properties) throws CalculationException, SolverException{
		final AbstractRandomVariableFactory randomVariableFactory = (AbstractRandomVariableFactory) properties.getOrDefault("RandomVariableFactory", new RandomVariableFactory());
		final ValueUnit valueUnit 						= (ValueUnit) 		properties.getOrDefault(	"ValueUnit", ValueUnit.VOLATILITYNORMAL);
		final OptimizerType optimizerType 				= (OptimizerType) 	properties.getOrDefault(	"OptimizerType", OptimizerType.LevenbergMarquardt);
		final OptimizerDerivativeType derivativeType 	= (OptimizerDerivativeType) properties.getOrDefault(	"DerivativeType", OptimizerDerivativeType.FiniteDifferences); 
		final OptimizerSolverType solverType 			= (OptimizerSolverType) 	properties.getOrDefault(	"SolverType", OptimizerSolverType.Vector); 
		final int maxIterations 						= (int) properties.getOrDefault("maxIterations", 400); 
		final long maxRunTimeInMillis 	= (long) 	properties.getOrDefault(	"maxRunTime", (long)6E5 /*10min*/); 
		final double errorTolerance 	= (double) 	properties.getOrDefault(	"errorTolerance", 0.0);
		final int numberOfThreads 		= (int) 	properties.getOrDefault(	"numberOfThreads", 2);
		final int numberOfFactors 		= (int) 	properties.getOrDefault(	"numberOfFactors", 1);
		final int numberOfPaths 		= (int) 	properties.getOrDefault(	"numberOfPathsSwaptionSmile", (int)1E5);
		final int seed 					= (int) 	properties.getOrDefault(	"seed", 1234);

		OptimizerFactoryInterface optimizerFactory = (OptimizerFactoryInterface) properties.get("OptimizerFactory");
		if(optimizerFactory == null){
			Map<String, Object> optimizerProperties = new HashMap<>();
			optimizerProperties.putAll(properties);
			optimizerProperties.putIfAbsent("maxIterations", 	maxIterations);
			optimizerProperties.putIfAbsent("maxRunTime", 		maxRunTimeInMillis);
			optimizerProperties.putIfAbsent("errorTolerance", 	errorTolerance);
			optimizerFactory = new OptimizerFactory(optimizerType, maxIterations, errorTolerance, numberOfThreads, optimizerProperties, true);
		}

		// print current configuration
		printConfigurations(valueUnit, optimizerType, derivativeType, solverType, maxIterations, maxRunTimeInMillis, errorTolerance, numberOfThreads, numberOfPaths, seed, numberOfFactors);
		
		/*
		 * Calibration test
		 */
		System.out.println("Calibration to Swaption Smile Products.");

		/*
		 * Definition of curves
		 */
		double[] fixingTimes = new double[] {
				0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.5, 50.0
		};

		double[] forwardRates = new double[] {
				0.61/100.0,	0.61/100.0,	0.67/100.0,	0.73/100.0,	0.80/100.0,	0.92/100.0,	1.11/100.0,	1.36/100.0,	1.60/100.0,	1.82/100.0,	2.02/100.0,	2.17/100.0,	2.27/100.0,	2.36/100.0,	2.46/100.0,	2.52/100.0,	2.54/100.0,	2.57/100.0,	2.68/100.0,	2.82/100.0,	2.92/100.0,	2.98/100.0,	3.00/100.0,	2.99/100.0,	2.95/100.0,	2.89/100.0,	2.82/100.0,	2.74/100.0,	2.66/100.0,	2.59/100.0,	2.52/100.0,	2.47/100.0,	2.42/100.0,	2.38/100.0,	2.35/100.0,	2.33/100.0,	2.31/100.0,	2.30/100.0,	2.29/100.0,	2.28/100.0,	2.27/100.0,	2.27/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.27/100.0,	2.28/100.0,	2.28/100.0,	2.30/100.0,	2.31/100.0,	2.32/100.0,	2.34/100.0,	2.35/100.0,	2.37/100.0,	2.39/100.0,	2.42/100.0,	2.44/100.0,	2.47/100.0,	2.50/100.0,	2.52/100.0,	2.56/100.0,	2.59/100.0,	2.62/100.0,	2.65/100.0,	2.68/100.0,	2.72/100.0,	2.75/100.0,	2.78/100.0,	2.81/100.0,	2.83/100.0,	2.86/100.0,	2.88/100.0,	2.91/100.0,	2.93/100.0,	2.94/100.0,	2.96/100.0,	2.97/100.0,	2.97/100.0,	2.97/100.0,	2.97/100.0,	2.97/100.0,	2.96/100.0,	2.95/100.0,	2.94/100.0,	2.93/100.0,	2.91/100.0,	2.89/100.0,	2.87/100.0,	2.85/100.0,	2.83/100.0,	2.80/100.0,	2.78/100.0,	2.75/100.0,	2.72/100.0,	2.69/100.0,	2.67/100.0,	2.64/100.0,	2.64/100.0
		};

		double liborPeriodLength = 0.5;

		// Create the forward curve (initial value of the LIBOR market model)
		ForwardCurve forwardCurve = ForwardCurve.createForwardCurveFromForwards(
				"forwardCurve"		/* name of the curve */,
				fixingTimes			/* fixings of the forward */,
				forwardRates		/* forwards */,
				liborPeriodLength	/* tenor / period length */
				);


		DiscountCurveInterface discountCurve = new DiscountCurveFromForwardCurve(forwardCurve, liborPeriodLength);
		
		/*
		 * Create a set of calibration products.
		 */
		ArrayList<String>					calibrationItemNames						= new ArrayList<String>();
		final ArrayList<CalibrationItem>	calibrationItemsVALUE						= new ArrayList<CalibrationItem>();
		final ArrayList<CalibrationItem>	calibrationItemsVOLATILITYNORMAL			= new ArrayList<CalibrationItem>();

		double	swapPeriodLength	= 0.5;
		int		numberOfPeriods		= 20;

		double[] smileMoneynesses	= { -0.02,	-0.01, -0.005, -0.0025,	0.0,	0.0025,	0.0050,	0.01,	0.02 };
		double[] smileVolatilities	= { 0.559,	0.377,	0.335,	 0.320,	0.308, 0.298, 0.290, 0.280, 0.270 };

		for(int i=0; i<smileMoneynesses.length; i++ ) {
			double	exerciseDate		= 5.0;
			double	moneyness			= smileMoneynesses[i];
			double	targetVolatility	= smileVolatilities[i];
			double	swapTenor			= numberOfPeriods*swapPeriodLength;
			
			double	weight = 1.0;
			
			calibrationItemsVALUE.add(createCalibrationItem(weight, exerciseDate, swapPeriodLength, numberOfPeriods, moneyness, targetVolatility, forwardCurve, discountCurve, ValueUnit.VALUE));
			calibrationItemsVOLATILITYNORMAL.add(createCalibrationItem(weight, exerciseDate, swapPeriodLength, numberOfPeriods, moneyness, targetVolatility, forwardCurve, discountCurve, ValueUnit.VOLATILITYLOGNORMAL));
		
			calibrationItemNames.add(exerciseDate+"\t"+swapTenor+"\t"+moneyness);
		}

		
		double[] atmOptionMaturities	= { 2.00, 3.00, 4.00, 5.00, 7.00, 10.00, 15.00, 20.00, 25.00, 30.00 };
		double[] atmOptionVolatilities	= { 0.385, 0.351, 0.325, 0.308, 0.288, 0.279, 0.290, 0.272, 0.235, 0.192 };

		for(int i=0; i<atmOptionMaturities.length; i++ ) {

			double	exerciseDate		= atmOptionMaturities[i];
			double	moneyness			= 0.0;
			double	targetVolatility	= atmOptionVolatilities[i];
			double	swapTenor			= numberOfPeriods*swapPeriodLength;

			double	weight = 1.0;
			
			calibrationItemsVALUE.add(createCalibrationItem(weight, exerciseDate, swapPeriodLength, numberOfPeriods, moneyness, targetVolatility, forwardCurve, discountCurve, ValueUnit.VALUE));
			calibrationItemsVOLATILITYNORMAL.add(createCalibrationItem(weight, exerciseDate, swapPeriodLength, numberOfPeriods, moneyness, targetVolatility, forwardCurve, discountCurve, ValueUnit.VOLATILITYLOGNORMAL));
			
			calibrationItemNames.add(exerciseDate+"\t"+swapTenor+"\t"+moneyness);
		}
		
		/*
		 * Create a LIBOR Market Model
		 */

		/*
		 * Create the libor tenor structure and the initial values
		 */
		double liborRateTimeHorzion	= 20.0;
		TimeDiscretization liborPeriodDiscretization = new TimeDiscretization(0.0, (int) (liborRateTimeHorzion / liborPeriodLength), liborPeriodLength);

		/*
		 * Create a simulation time discretization
		 */
		double lastTime	= 20.0;
		double dt		= 0.5;
		TimeDiscretization timeDiscretization = new TimeDiscretization(0.0, (int) (lastTime / dt), dt);

		/*
		 * Create Brownian motions 
		 */
		final BrownianMotionInterface brownianMotion = new net.finmath.montecarlo.BrownianMotion(timeDiscretization, 6, numberOfPaths, seed);
		final BrownianMotionInterface brownianMotionSim = new BrownianMotionView(brownianMotion, new Integer[] { 0, 1, 2, 3, 4 });
		final BrownianMotionInterface brownianMotionView2 = new BrownianMotionView(brownianMotion, new Integer[] { 0, 5 });

		TimeDiscretizationInterface volatilitySurfaceDiscretization = new TimeDiscretization(0.00, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0); 
		double[] initialVolatility = new double[] { 0.50 / 100 };
		LIBORVolatilityModel volatilityModel = new LIBORVolatilityModelPiecewiseConstant(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, volatilitySurfaceDiscretization, volatilitySurfaceDiscretization, initialVolatility, true);
		/* Correlation Model with exponential decay */
		LIBORCorrelationModel correlationModel = new LIBORCorrelationModelExponentialDecay(timeDiscretization, liborPeriodDiscretization, 5, 0.05, false);

		// Create a covariance model
		//AbstractLIBORCovarianceModelParametric covarianceModelParametric = new LIBORCovarianceModelExponentialForm5Param(timeDiscretization, liborPeriodDiscretization, numberOfFactors, new double[] { 0.20/100.0, 0.05/100.0, 0.10, 0.05/100.0, 0.10} );
		AbstractLIBORCovarianceModelParametric covarianceModelParametric1 = new LIBORCovarianceModelFromVolatilityAndCorrelation(timeDiscretization, liborPeriodDiscretization, volatilityModel, correlationModel);
		AbstractLIBORCovarianceModelParametric covarianceModelParametric = new LIBORCovarianceModelStochasticVolatility(randomVariableFactory, covarianceModelParametric1, brownianMotionView2, 0.01, -0.30, true);

		
		// Set model properties
		Map<String, Object> calibrationProperties = new HashMap<String, Object>();

		// Choose the simulation measure
		calibrationProperties.put("measure", LIBORMarketModel.Measure.SPOT.name());

		// Choose normal state space for the Euler scheme (the covariance model above carries a linear local volatility model, such that the resulting model is log-normal).
		calibrationProperties.put("stateSpace", LIBORMarketModel.StateSpace.NORMAL.name());
		
		/*
		 * The optimizer to use and some of its parameters
		 */

		Map<String, Object> calibrationParameters = new HashMap<String, Object>();
		calibrationParameters.put("brownianMotion", 	brownianMotionSim);
		calibrationParameters.put("optimizerFactory", 	optimizerFactory);
		calibrationParameters.put("parameterStep", 		new Double(1E-4));

		calibrationParameters.put("solverType", 		solverType);
		calibrationParameters.put("derivativeType", 	derivativeType);
		calibrationParameters.put("scheme", 			Scheme.EULER);

		calibrationProperties.put("calibrationParameters", calibrationParameters);
		
		
		LIBORMarketModel.CalibrationItem[] calibrationItemsLMM = new LIBORMarketModel.CalibrationItem[calibrationItemNames.size()];
		for(int i=0; i<calibrationItemNames.size(); i++){
			CalibrationItem calibrationItem = (valueUnit == ValueUnit.VALUE) ? calibrationItemsVALUE.get(i) : calibrationItemsVOLATILITYNORMAL.get(i);
			calibrationItemsLMM[i] = new LIBORMarketModel.CalibrationItem(calibrationItem.calibrationProduct,calibrationItem.calibrationTargetValue,calibrationItem.calibrationWeight);
		}		
			
		LIBORMarketModel liborMarketModelCalibrated = new LIBORMarketModel(
				liborPeriodDiscretization,
				null,
				forwardCurve, discountCurve, 
				randomVariableFactory, 
				covarianceModelParametric,
				calibrationItemsLMM, calibrationProperties);	
		
		evaluateCalibration(liborMarketModelCalibrated, brownianMotionSim,
				calibrationItemNames, calibrationItemsVALUE, calibrationItemsVOLATILITYNORMAL,
				(OptimizerInterfaceAAD) covarianceModelParametric.getCalibrationOptimizer(),
				2E-2 /*assertTrueVALUE*/, 1E-2 /*assertTrueVOLATILITYNORMAL*/,
				"SwaptionSmile" + "-" + derivativeType + "-" + optimizerType + "-" + valueUnit + "-" + numberOfPaths);
	}
	
	
	//------------------------------------------------------------------------------------------
	public static void ATMSwaptionCalibration(Map<String, Object> properties) throws CalculationException, SolverException{
		final AbstractRandomVariableFactory randomVariableFactory = (AbstractRandomVariableFactory) properties.getOrDefault("RandomVariableFactory", new RandomVariableFactory());
		final ValueUnit valueUnit 						= (ValueUnit) 		properties.getOrDefault(	"ValueUnit", ValueUnit.VOLATILITYNORMAL);
		final OptimizerType optimizerType 				= (OptimizerType) 	properties.getOrDefault(	"OptimizerType", OptimizerType.LevenbergMarquardt);
		final OptimizerDerivativeType derivativeType 	= (OptimizerDerivativeType) properties.getOrDefault(	"DerivativeType", OptimizerDerivativeType.FiniteDifferences); 
		final OptimizerSolverType solverType 			= (OptimizerSolverType) 	properties.getOrDefault(	"SolverType", OptimizerSolverType.Vector); 
		final int maxIterations 						= (int) properties.getOrDefault("maxIterations", 400); 
		final long maxRunTimeInMillis 	= (long) 	properties.getOrDefault(	"maxRunTime", (long)6E5 /*10min*/); 
		final double errorTolerance 	= (double) 	properties.getOrDefault(	"errorTolerance", 0.0);
		final int numberOfThreads 		= (int) 	properties.getOrDefault(	"numberOfThreads", 2);
		final int numberOfFactors 		= (int) 	properties.getOrDefault(	"numberOfFactors", 1);
		final int numberOfPaths 		= (int) 	properties.getOrDefault(	"numberOfPathsATM", (int)1E3);
		final int seed 					= (int) 	properties.getOrDefault(	"seed", 1234);

		OptimizerFactoryInterface optimizerFactory = (OptimizerFactoryInterface) properties.get("OptimizerFactory");
		if(optimizerFactory == null){
			Map<String, Object> optimizerProperties = new HashMap<>();
			optimizerProperties.putAll(properties);
			optimizerProperties.putIfAbsent("maxIterations", 	maxIterations);
			optimizerProperties.putIfAbsent("maxRunTime", 		maxRunTimeInMillis);
			optimizerProperties.putIfAbsent("errorTolerance", 	errorTolerance);
			optimizerFactory = new OptimizerFactory(optimizerType, maxIterations, errorTolerance, numberOfThreads, optimizerProperties, true);
		}

		// print current configuration
		printConfigurations(valueUnit, optimizerType, derivativeType, solverType, maxIterations, maxRunTimeInMillis, errorTolerance, numberOfThreads, numberOfPaths, seed, numberOfFactors);

		/*
		 * Calibration test
		 */
		System.out.println("Calibration to ATM Swaptions.");

		/*
		 * Calibration of rate curves
		 */
		
		final AnalyticModelInterface curveModel = getCalibratedCurve();

		// Create the forward curve (initial value of the LIBOR market model)
		final ForwardCurveInterface forwardCurve = curveModel.getForwardCurve("ForwardCurveFromDiscountCurve(discountCurve-EUR,6M)");
		final DiscountCurveInterface discountCurve = curveModel.getDiscountCurve("discountCurve-EUR");
		//		curveModel.addCurve(discountCurve.getName(), discountCurve);

		//		long millisCurvesEnd = System.currentTimeMillis();
		//		System.out.println("");

		/*
		 * Calibration of model volatilities
		 */
		System.out.println("Brute force Monte-Carlo calibration of model volatilities:");

		/*
		 * Create a set of calibration products.
		 */
		ArrayList<String>					calibrationItemNames		= new ArrayList<String>();
		final ArrayList<CalibrationItem>	calibrationItemsVALUE						= new ArrayList<CalibrationItem>();
		final ArrayList<CalibrationItem>	calibrationItemsVOLATILITYNORMAL			= new ArrayList<CalibrationItem>();

		double	swapPeriodLength	= 0.5;

		String[] atmExpiries = { "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "1M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "3M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "6M", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "1Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "2Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "3Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "4Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "5Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "7Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "10Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "15Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "20Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "25Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y", "30Y" };
		String[] atmTenors = { "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y" };
		double[] atmNormalVolatilities = { 0.00151, 0.00169, 0.0021, 0.00248, 0.00291, 0.00329, 0.00365, 0.004, 0.00437, 0.00466, 0.00527, 0.00571, 0.00604, 0.00625, 0.0016, 0.00174, 0.00217, 0.00264, 0.00314, 0.00355, 0.00398, 0.00433, 0.00469, 0.00493, 0.00569, 0.00607, 0.00627, 0.00645, 0.00182, 0.00204, 0.00238, 0.00286, 0.00339, 0.00384, 0.00424, 0.00456, 0.00488, 0.0052, 0.0059, 0.00623, 0.0064, 0.00654, 0.00205, 0.00235, 0.00272, 0.0032, 0.00368, 0.00406, 0.00447, 0.00484, 0.00515, 0.00544, 0.00602, 0.00629, 0.0064, 0.00646, 0.00279, 0.00319, 0.0036, 0.00396, 0.00436, 0.00469, 0.00503, 0.0053, 0.00557, 0.00582, 0.00616, 0.00628, 0.00638, 0.00641, 0.00379, 0.00406, 0.00439, 0.00472, 0.00504, 0.00532, 0.0056, 0.00582, 0.00602, 0.00617, 0.0063, 0.00636, 0.00638, 0.00639, 0.00471, 0.00489, 0.00511, 0.00539, 0.00563, 0.00583, 0.006, 0.00618, 0.0063, 0.00644, 0.00641, 0.00638, 0.00635, 0.00634, 0.00544, 0.00557, 0.00572, 0.00591, 0.00604, 0.00617, 0.0063, 0.00641, 0.00651, 0.00661, 0.00645, 0.00634, 0.00627, 0.00624, 0.00625, 0.00632, 0.00638, 0.00644, 0.0065, 0.00655, 0.00661, 0.00667, 0.00672, 0.00673, 0.00634, 0.00614, 0.00599, 0.00593, 0.00664, 0.00671, 0.00675, 0.00676, 0.00676, 0.00675, 0.00676, 0.00674, 0.00672, 0.00669, 0.00616, 0.00586, 0.00569, 0.00558, 0.00647, 0.00651, 0.00651, 0.00651, 0.00652, 0.00649, 0.00645, 0.0064, 0.00637, 0.00631, 0.00576, 0.00534, 0.00512, 0.00495, 0.00615, 0.0062, 0.00618, 0.00613, 0.0061, 0.00607, 0.00602, 0.00596, 0.00591, 0.00586, 0.00536, 0.00491, 0.00469, 0.0045, 0.00578, 0.00583, 0.00579, 0.00574, 0.00567, 0.00562, 0.00556, 0.00549, 0.00545, 0.00538, 0.00493, 0.00453, 0.00435, 0.0042, 0.00542, 0.00547, 0.00539, 0.00532, 0.00522, 0.00516, 0.0051, 0.00504, 0.005, 0.00495, 0.00454, 0.00418, 0.00404, 0.00394 };

		LocalDate referenceDate = LocalDate.of(2016, Month.SEPTEMBER, 30); 
		BusinessdayCalendarExcludingTARGETHolidays cal = new BusinessdayCalendarExcludingTARGETHolidays();
		DayCountConvention_ACT_365 modelDC = new DayCountConvention_ACT_365();
		for(int i=0; i<atmNormalVolatilities.length; i++ ) {

			LocalDate exerciseDate = cal.createDateFromDateAndOffsetCode(referenceDate, atmExpiries[i]);
			LocalDate tenorEndDate = cal.createDateFromDateAndOffsetCode(exerciseDate, atmTenors[i]);
			double	exercise		= modelDC.getDaycountFraction(referenceDate, exerciseDate);
			double	tenor			= modelDC.getDaycountFraction(exerciseDate, tenorEndDate);

			// We consider an idealized tenor grid (alternative: adapt the model grid)
			exercise	= Math.round(exercise/0.25)*0.25;
			tenor		= Math.round(tenor/0.25)*0.25;

			if(exercise < 1.0) continue;

			int numberOfPeriods = (int)Math.round(tenor / swapPeriodLength);

			double	moneyness			= 0.0;
			double	targetVolatility	= atmNormalVolatilities[i];

			double	weight = 1.0;

			calibrationItemsVALUE.add(createATMCalibrationItem(weight, exercise, swapPeriodLength, numberOfPeriods, moneyness, targetVolatility, forwardCurve, discountCurve, ValueUnit.VALUE));
			calibrationItemsVOLATILITYNORMAL.add(createATMCalibrationItem(weight, exercise, swapPeriodLength, numberOfPeriods, moneyness, targetVolatility, forwardCurve, discountCurve, ValueUnit.VOLATILITYNORMAL));

			calibrationItemNames.add(atmExpiries[i]+"\t"+atmTenors[i]);
		}

		/*
		 * Create a simulation time discretization
		 */
		// If simulation time is below libor time, exceptions will be hard to track.
		double lastTime	= 40.0;
		double dt		= 0.25;
		TimeDiscretization timeDiscretization = new TimeDiscretization(0.0, (int) (lastTime / dt), dt);
		final TimeDiscretizationInterface liborPeriodDiscretization = timeDiscretization;

		/*
		 * Create Brownian motions 
		 */
		final BrownianMotionInterface brownianMotion = new net.finmath.montecarlo.BrownianMotion(timeDiscretization, numberOfFactors, numberOfPaths, seed);
					
		AbstractLIBORCovarianceModelParametric covarianceModelParametric = createInitialCovarianceModel(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, numberOfFactors);

		// Set model properties
		Map<String, Object> calibrtionProperties = new HashMap<String, Object>();

		// Choose the simulation measure
		calibrtionProperties.put("measure", LIBORMarketModel.Measure.SPOT.name());

		// Choose normal state space for the Euler scheme (the covariance model above carries a linear local volatility model, such that the resulting model is log-normal).
		calibrtionProperties.put("stateSpace", LIBORMarketModel.StateSpace.NORMAL.name());

		double[] parameterStandardDeviation = new double[covarianceModelParametric.getParameter().length];
		double[] parameterLowerBound = new double[covarianceModelParametric.getParameter().length];
		double[] parameterUpperBound = new double[covarianceModelParametric.getParameter().length];
		Arrays.fill(parameterStandardDeviation, 0.20/100.0);
		Arrays.fill(parameterLowerBound, 0.0);
		Arrays.fill(parameterUpperBound, Double.POSITIVE_INFINITY);


		// Set calibration calibrtionProperties (should use our brownianMotion for calibration - needed to have to right correlation).
		Map<String, Object> calibrationParameters = new HashMap<String, Object>();
		calibrationParameters.put("numberOfThreads", numberOfThreads);

		calibrationParameters.put("brownianMotion", brownianMotion);
		calibrationParameters.put("optimizerFactory", optimizerFactory);
		calibrationParameters.put("parameterStep", new Double(1E-4));

		calibrationParameters.put("solverType", solverType);
		calibrationParameters.put("derivativeType", derivativeType);
		calibrationParameters.put("scheme", Scheme.EULER);
		calibrationParameters.put("parameterLowerBound", parameterLowerBound);
		calibrationParameters.put("parameterUpperBound", parameterUpperBound);

		calibrtionProperties.put("calibrationParameters", calibrationParameters);

		/*
		 * Create corresponding LIBOR Market Model
		 */
		LIBORMarketModel.CalibrationItem[] calibrationItemsLMM = new LIBORMarketModel.CalibrationItem[calibrationItemNames.size()];
		for(int i=0; i<calibrationItemNames.size(); i++){
			CalibrationItem calibrationItem = valueUnit == ValueUnit.VALUE ? calibrationItemsVALUE.get(i) : calibrationItemsVOLATILITYNORMAL.get(i);
			calibrationItemsLMM[i] = new LIBORMarketModel.CalibrationItem(calibrationItem.calibrationProduct,calibrationItem.calibrationTargetValue,calibrationItem.calibrationWeight);
		}
		LIBORModelInterface liborMarketModelCalibrated = new LIBORMarketModel(
				liborPeriodDiscretization,
				curveModel,
				forwardCurve, new DiscountCurveFromForwardCurve(forwardCurve),
				randomVariableFactory,
				covarianceModelParametric,
				calibrationItemsLMM,
				calibrtionProperties);
		
		evaluateCalibration(liborMarketModelCalibrated, brownianMotion,
				calibrationItemNames, calibrationItemsVALUE, calibrationItemsVOLATILITYNORMAL,
				(OptimizerInterfaceAAD) covarianceModelParametric.getCalibrationOptimizer(), 
				2E-4 /*assertTrueVALUE*/, 1E-4 /*assertTrueVOLATILITYNORMAL*/, 
				"ATM" + "-" + derivativeType + "-" + optimizerType + "-" + valueUnit + "-" + numberOfPaths);		
	}

	public static AnalyticModelInterface getCalibratedCurve() throws SolverException {
		final String[] maturity					= { "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "11Y", "12Y", "15Y", "20Y", "25Y", "30Y", "35Y", "40Y", "45Y", "50Y" };
		final String[] frequency				= { "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual", "annual" };
		final String[] frequencyFloat			= { "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual", "semiannual" };
		final String[] daycountConventions		= { "ACT/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360", "E30/360" };
		final String[] daycountConventionsFloat	= { "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360", "ACT/360" };
		final double[] rates					= { -0.00216 ,-0.00208 ,-0.00222 ,-0.00216 ,-0.0019 ,-0.0014 ,-0.00072 ,0.00011 ,0.00103 ,0.00196 ,0.00285 ,0.00367 ,0.0044 ,0.00604 ,0.00733 ,0.00767 ,0.00773 ,0.00765 ,0.00752 ,0.007138 ,0.007 };

		HashMap<String, Object> parameters = new HashMap<String, Object>();

		parameters.put("referenceDate", LocalDate.of(2016, Month.SEPTEMBER, 30)); 
		parameters.put("currency", "EUR");
		parameters.put("forwardCurveTenor", "6M");
		parameters.put("maturities", maturity);
		parameters.put("fixLegFrequencies", frequency);
		parameters.put("floatLegFrequencies", frequencyFloat);
		parameters.put("fixLegDaycountConventions", daycountConventions);
		parameters.put("floatLegDaycountConventions", daycountConventionsFloat);
		parameters.put("rates", rates);

		return getCalibratedCurve(null, parameters);
	}

	private static AnalyticModelInterface getCalibratedCurve(AnalyticModelInterface model2, Map<String, Object> parameters) throws SolverException {

		final LocalDate	referenceDate		= (LocalDate) parameters.get("referenceDate");
		final String	currency			= (String) parameters.get("currency");
		final String	forwardCurveTenor	= (String) parameters.get("forwardCurveTenor");
		final String[]	maturities			= (String[]) parameters.get("maturities");
		final String[]	frequency			= (String[]) parameters.get("fixLegFrequencies");
		final String[]	frequencyFloat		= (String[]) parameters.get("floatLegFrequencies");
		final String[]	daycountConventions	= (String[]) parameters.get("fixLegDaycountConventions");
		final String[]	daycountConventionsFloat	= (String[]) parameters.get("floatLegDaycountConventions");
		final double[]	rates						= (double[]) parameters.get("rates");

		Assert.assertEquals(maturities.length, frequency.length);
		Assert.assertEquals(maturities.length, daycountConventions.length);
		Assert.assertEquals(maturities.length, rates.length);

		Assert.assertEquals(frequency.length, frequencyFloat.length);
		Assert.assertEquals(daycountConventions.length, daycountConventionsFloat.length);

		int		spotOffsetDays = 2;
		String	forwardStartPeriod = "0D";

		String curveNameDiscount = "discountCurve-" + currency;

		/*
		 * We create a forward curve by referencing the same discount curve, since
		 * this is a single curve setup.
		 * 
		 * Note that using an independent NSS forward curve with its own NSS parameters
		 * would result in a problem where both, the forward curve and the discount curve
		 * have free parameters.
		 */
		ForwardCurveInterface forwardCurve		= new ForwardCurveFromDiscountCurve(curveNameDiscount, referenceDate, forwardCurveTenor);

		// Create a collection of objective functions (calibration products)
		Vector<AnalyticProductInterface> calibrationProducts = new Vector<AnalyticProductInterface>();
		double[] curveMaturities	= new double[rates.length+1];
		double[] curveValue			= new double[rates.length+1];
		boolean[] curveIsParameter	= new boolean[rates.length+1];
		curveMaturities[0] = 0.0;
		curveValue[0] = 1.0;
		curveIsParameter[0] = false;
		for(int i=0; i<rates.length; i++) {

			ScheduleInterface schedulePay = ScheduleGenerator.createScheduleFromConventions(referenceDate, spotOffsetDays, forwardStartPeriod, maturities[i], frequency[i], daycountConventions[i], "first", "following", new BusinessdayCalendarExcludingTARGETHolidays(), -2, 0);
			ScheduleInterface scheduleRec = ScheduleGenerator.createScheduleFromConventions(referenceDate, spotOffsetDays, forwardStartPeriod, maturities[i], frequencyFloat[i], daycountConventionsFloat[i], "first", "following", new BusinessdayCalendarExcludingTARGETHolidays(), -2, 0);

			curveMaturities[i+1] = Math.max(schedulePay.getPayment(schedulePay.getNumberOfPeriods()-1),scheduleRec.getPayment(scheduleRec.getNumberOfPeriods()-1));
			curveValue[i+1] = 1.0;
			curveIsParameter[i+1] = true;
			calibrationProducts.add(new Swap(schedulePay, null, rates[i], curveNameDiscount, scheduleRec, forwardCurve.getName(), 0.0, curveNameDiscount));
		}

		InterpolationMethod interpolationMethod = InterpolationMethod.LINEAR;

		// Create a discount curve
		DiscountCurve			discountCurve					= DiscountCurve.createDiscountCurveFromDiscountFactors(
				curveNameDiscount								/* name */,
				curveMaturities	/* maturities */,
				curveValue		/* discount factors */,
				curveIsParameter,
				interpolationMethod ,
				ExtrapolationMethod.CONSTANT,
				InterpolationEntity.LOG_OF_VALUE
				);

		/*
		 * Model consists of the two curves, but only one of them provides free parameters.
		 */
		AnalyticModelInterface model = new AnalyticModel(new CurveInterface[] { discountCurve, forwardCurve });

		/*
		 * Create a collection of curves to calibrate
		 */
		Set<ParameterObjectInterface> curvesToCalibrate = new HashSet<ParameterObjectInterface>();
		curvesToCalibrate.add(discountCurve);

		/*
		 * Calibrate the curve
		 */
		Solver solver = new Solver(model, calibrationProducts, 0.0, 1E-4 /* target accuracy */);
		AnalyticModelInterface calibratedModel = solver.getCalibratedModel(curvesToCalibrate);
		//		System.out.println("Solver reported acccurary....: " + solver.getAccuracy());

		Assert.assertEquals("Calibration accurarcy", 0.0, solver.getAccuracy(), 1E-3);

		model			= calibratedModel;
	
		return model;
	}

	private static double getParSwaprate(ForwardCurveInterface forwardCurve, DiscountCurveInterface discountCurve, double[] swapTenor) throws CalculationException {
		return net.finmath.marketdata.products.Swap.getForwardSwapRate(new TimeDiscretization(swapTenor), new TimeDiscretization(swapTenor), forwardCurve, discountCurve);
	}

	private static CalibrationItem createATMCalibrationItem(double weight, double exerciseDate, double swapPeriodLength, int numberOfPeriods, double moneyness, double targetVolatility, ForwardCurveInterface forwardCurve, DiscountCurveInterface discountCurve, ValueUnit valueUnit) throws CalculationException {

		double[]	fixingDates			= new double[numberOfPeriods];
		double[]	paymentDates		= new double[numberOfPeriods];
		double[]	swapTenor			= new double[numberOfPeriods + 1];
		for (int periodStartIndex = 0; periodStartIndex < numberOfPeriods; periodStartIndex++) {
			fixingDates[periodStartIndex] = exerciseDate + periodStartIndex * swapPeriodLength;
			paymentDates[periodStartIndex] = exerciseDate + (periodStartIndex + 1) * swapPeriodLength;
			swapTenor[periodStartIndex] = exerciseDate + periodStartIndex * swapPeriodLength;
		}
		swapTenor[numberOfPeriods] = exerciseDate + numberOfPeriods * swapPeriodLength;

		AbstractLIBORMonteCarloProduct swaptionMonteCarlo = new ATMSwaption(swapTenor, valueUnit);

		/*
		 * We use Monte-Carlo calibration on implied volatility.
		 * Alternatively you may change here to Monte-Carlo valuation on price or
		 * use an analytic approximation formula, etc.
		 */
		CalibrationItem calibrationItem = null;
		switch(valueUnit) {
		case VALUE:
			double swaprate = moneyness + getParSwaprate(forwardCurve, discountCurve, swapTenor);
			double swapannuity = SwapAnnuity.getSwapAnnuity(new TimeDiscretization(swapTenor), discountCurve);
			double targetPrice = AnalyticFormulas.bachelierOptionValue(
					new RandomVariable(swaprate),
					new RandomVariable(targetVolatility), swapTenor[0], swaprate, 
					new RandomVariable(swapannuity)).doubleValue();
			calibrationItem  = new CalibrationItem(swaptionMonteCarlo, targetPrice, weight);
			break;
		case INTEGRATEDNORMALVARIANCE:
			targetVolatility = targetVolatility * targetVolatility * swapTenor[0];
		case VOLATILITYNORMAL:
			calibrationItem = new CalibrationItem(swaptionMonteCarlo, targetVolatility, weight);
			break;
		default:
			throw new UnsupportedOperationException();
		}

		return calibrationItem;
	}

	private static CalibrationItem createCalibrationItem(double weight, double exerciseDate, double swapPeriodLength, int numberOfPeriods, double moneyness, double targetVolatility, ForwardCurveInterface forwardCurve, DiscountCurveInterface discountCurve, ValueUnit valueUnit) throws CalculationException {

		double[]	fixingDates			= new double[numberOfPeriods];
		double[]	paymentDates		= new double[numberOfPeriods];
		double[]	swapTenor			= new double[numberOfPeriods + 1];

		for (int periodStartIndex = 0; periodStartIndex < numberOfPeriods; periodStartIndex++) {
			fixingDates[periodStartIndex] = exerciseDate + periodStartIndex * swapPeriodLength;
			paymentDates[periodStartIndex] = exerciseDate + (periodStartIndex + 1) * swapPeriodLength;
			swapTenor[periodStartIndex] = exerciseDate + periodStartIndex * swapPeriodLength;
		}
		swapTenor[numberOfPeriods] = exerciseDate + numberOfPeriods * swapPeriodLength;

		// Swaptions swap rate
		double swaprate = moneyness + getParSwaprate(forwardCurve, discountCurve, swapTenor);

		// Set swap rates for each period
		double[] swaprates = new double[numberOfPeriods];
		Arrays.fill(swaprates, swaprate);

		SwaptionSimple swaptionMonteCarlo = new SwaptionSimple(swaprate, swapTenor, valueUnit);
		
		/*
		 * We use Monte-Carlo calibration on implied volatility.
		 * Alternatively you may change here to Monte-Carlo valuation on price or
		 * use an analytic approximation formula, etc.
		 */
		CalibrationItem calibrationItem = null;
		switch (valueUnit) {
		case VALUE:
			double targetValuePrice = AnalyticFormulas.blackModelSwaptionValue(
					swaprate, targetVolatility, 
					fixingDates[0], swaprate, 
					SwapAnnuity.getSwapAnnuity(new TimeDiscretization(swapTenor), discountCurve));
			calibrationItem = new CalibrationItem(swaptionMonteCarlo, targetValuePrice, weight);
			break;
		case INTEGRATEDLOGNORMALVARIANCE:
			targetVolatility = targetVolatility * targetVolatility * swapTenor[0];
		case VOLATILITYLOGNORMAL:
			calibrationItem = new CalibrationItem(swaptionMonteCarlo, targetVolatility, weight);
			break;
		default:
			throw new UnsupportedOperationException();
		}
		
		return calibrationItem;
	}

	private static void evaluateCalibration(LIBORModelInterface liborMarketModelCalibrated, BrownianMotionInterface brownianMotion, 
			List<String> calibrationItemNames ,List<CalibrationItem> calibrationItemsVALUE, List<CalibrationItem> calibrationItemsVOLATILITYNORMAL,
			OptimizerInterfaceAAD optimizer,
			double assertTrueVALUE, double assertTrueVOLATILITYNORMAL, String FileName){
		
		if(FileName != null) {
			/* write calibration log to files*/
			String fileLocation = "calibration-logs/";
			String fileName = FileName + "-" + System.currentTimeMillis() + ".dat";
			FileManagement.writeOnCSVInitialize(false, fileLocation + fileName);
//			System.out.println(optimizer.getCalibrationLog());
			FileManagement.writeOnCSV(optimizer.getCalibrationLog());
			FileManagement.writeOnCSVflushclose();
		}
		
		System.out.println("\nCalibrated parameters are:");
		AbstractLIBORCovarianceModelParametric calibratedCovarianceModel = (AbstractLIBORCovarianceModelParametric) ((LIBORMarketModel) liborMarketModelCalibrated).getCovarianceModel();
		double[] param = calibratedCovarianceModel.getParameter();
		for (double p : param) 
			System.out.println(formatterParam.format(p));

		ProcessEulerScheme process = new ProcessEulerScheme(brownianMotion);
		LIBORModelMonteCarloSimulationInterface simulationCalibrated = new LIBORModelMonteCarloSimulation(liborMarketModelCalibrated, process);

		System.out.println("\nValuation on calibrated prices:");
		double deviationSumVALUE			= 0.0;
		double deviationSquaredSumVALUE	= 0.0;
		for (int i = 0; i < calibrationItemsVALUE.size(); i++) {
			CalibrationItem calibrationItem = calibrationItemsVALUE.get(i);
			try {
				double valueModel = calibrationItem.calibrationProduct.getValue(simulationCalibrated);
				double valueTarget = calibrationItem.calibrationTargetValue;
				double error = valueModel-valueTarget;
				deviationSumVALUE += error;
				deviationSquaredSumVALUE += error*error;
				System.out.println(calibrationItemNames.get(i) + "\t" + "Model: " + formatterValue.format(valueModel) + "\t Target: " + formatterValue.format(valueTarget) + "\t Deviation: " + formatterDeviation.format(valueModel-valueTarget));// + "\t" + calibrationProduct.toString());
			}
			catch(Exception e) {
			}
		}

		System.out.println("\nValuation on calibrated implieded Volatilities:");
		double deviationSumVOLATILITYNORMAL			= 0.0;
		double deviationSquaredSumVOLATILITYNORMAL	= 0.0;
		for (int i = 0; i < calibrationItemsVALUE.size(); i++) {
			CalibrationItem calibrationItem = calibrationItemsVOLATILITYNORMAL.get(i);
			try {
				double valueModel = calibrationItem.calibrationProduct.getValue(simulationCalibrated);
				double valueTarget = calibrationItem.calibrationTargetValue;
				double error = valueModel-valueTarget;
				deviationSumVOLATILITYNORMAL += error;
				deviationSquaredSumVOLATILITYNORMAL += error*error;
				System.out.println(calibrationItemNames.get(i) + "\t" + "Model: " + formatterValue.format(valueModel) + "\t Target: " + formatterValue.format(valueTarget) + "\t Deviation: " + formatterDeviation.format(valueModel-valueTarget));// + "\t" + calibrationProduct.toString());
			}
			catch(Exception e) {
			}
		}

		//		System.out.println("Calibration of curves........." + (millisCurvesEnd-millisCurvesStart)/1000.0);
		System.out.println();
		System.out.println("Calibration of volatilities..." + (optimizer.getRunTime()/1000.0) + "s");
		System.out.println("Number of Iterations.........." + optimizer.getIterations());

		double averageDeviationVALUE = deviationSumVALUE/calibrationItemsVALUE.size();
		double averageDeviationVOLATILITYNORMAL = deviationSumVOLATILITYNORMAL/calibrationItemsVOLATILITYNORMAL.size();

		System.out.println();
		System.out.println("Mean Deviation for prices........." + formatterValue.format(averageDeviationVALUE));
		System.out.println("Mean Deviation for volatilites...." + formatterValue.format(averageDeviationVOLATILITYNORMAL));
		System.out.println();
		System.out.println("RMS Error for prices.............." + formatterValue.format(Math.sqrt(deviationSquaredSumVALUE/calibrationItemsVALUE.size())));
		System.out.println("RMS Error for volatilites........." + formatterValue.format(Math.sqrt(deviationSquaredSumVOLATILITYNORMAL/calibrationItemsVOLATILITYNORMAL.size())));
		System.out.println("__________________________________________________________________________________________\n");

		// evaluate the two deviation averages
		Assert.assertEquals(0.0, averageDeviationVALUE, assertTrueVALUE);
		Assert.assertEquals(0.0, averageDeviationVOLATILITYNORMAL, assertTrueVOLATILITYNORMAL);
	}
	
	private static void printConfigurations(ValueUnit valueUnit, OptimizerType optimizerType, OptimizerDerivativeType derivativeType, OptimizerSolverType solverType, int maxIterations,
			long maxRunTimeInMillis, double errorTolerance, int numberOfThreads, int numberOfPaths, int seed, int numberOfFactors) {
		System.out.println("---------------------------------------------------------------");
		System.out.println("Configuration:");

		System.out.println("Value Unit.............." + valueUnit);
		System.out.println("Optimizer Type.........." + optimizerType);
		System.out.println("Derivative Type........." + derivativeType);
		System.out.println("Solver Type............." + solverType);	
		System.out.println("maxNumberOfIterations..." + maxIterations);
		System.out.println("maxRunTimeInMillis......" + maxRunTimeInMillis);
		System.out.println("errorTolerance.........." + errorTolerance);
		System.out.println("numberOfThreads........." + numberOfThreads);
		System.out.println("numberOfPaths..........." + numberOfPaths);
		System.out.println("seed...................." + seed);
		System.out.println("numberOfFactors........." + numberOfFactors);
		System.out.println();
		System.out.println("Current Time............" + LocalDateTime.now());
		System.out.println("maxSizeJVM.............." + (Runtime.getRuntime().maxMemory()/1024.0/1024.0) + " MB");
		System.out.println("numberOfCores..........." + Runtime.getRuntime().availableProcessors());
		System.out.println("---------------------------------------------------------------\n");
	}

	
	private static AbstractLIBORCovarianceModelParametric createInitialCovarianceModel(AbstractRandomVariableFactory randomVariableFactory, TimeDiscretizationInterface timeDiscretization, TimeDiscretizationInterface liborPeriodDiscretization, int numberOfFactors) {
		/* volatility model from piecewise constant interpolated matrix */
		TimeDiscretizationInterface volatilitySurfaceDiscretization = new TimeDiscretization(0.00, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0); 
		double[] initialVolatility = new double[] { 0.50 / 100 };
		LIBORVolatilityModel volatilityModel = new LIBORVolatilityModelPiecewiseConstant(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, volatilitySurfaceDiscretization, volatilitySurfaceDiscretization, initialVolatility, true);

		//		/* volatility model from given matrix */
		//		double initialVolatility = 0.005;
		//		double[][] volatility = new double[timeDiscretization.getNumberOfTimeSteps()][liborPeriodDiscretization.getNumberOfTimeSteps()];
		//		for(int i = 0; i < timeDiscretization.getNumberOfTimeSteps(); i++) Arrays.fill(volatility[i], initialVolatility);
		//		LIBORVolatilityModel volatilityModel = new LIBORVolatilityModelFromGivenMatrix(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, volatility);

		/* Correlation Model with exponential decay */
		LIBORCorrelationModel correlationModel = new LIBORCorrelationModelExponentialDecay(timeDiscretization, liborPeriodDiscretization, numberOfFactors, 0.05, false);

		// Create a covariance model
		AbstractLIBORCovarianceModelParametric covarianceModelParametric = new LIBORCovarianceModelFromVolatilityAndCorrelation(timeDiscretization, liborPeriodDiscretization, volatilityModel, correlationModel);

		// Create blended local volatility model with fixed parameter (0=lognormal, > 1 = almost a normal model).			
		AbstractLIBORCovarianceModelParametric covarianceModelDisplaced = new DisplacedLocalVolatilityModel(randomVariableFactory, covarianceModelParametric, 1.0/0.25, false /* isCalibrateable */);

		return covarianceModelDisplaced;
	}
}
