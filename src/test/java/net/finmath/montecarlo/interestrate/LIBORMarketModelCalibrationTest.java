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
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
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
import net.finmath.marketdata.model.curves.ForwardCurveFromDiscountCurve;
import net.finmath.marketdata.model.curves.ForwardCurveInterface;
import net.finmath.marketdata.products.AnalyticProductInterface;
import net.finmath.marketdata.products.Swap;
import net.finmath.marketdata.products.SwapAnnuity;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.BrownianMotionInterface;
import net.finmath.montecarlo.RandomVariable;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.automaticdifferentiation.backward.RandomVariableDifferentiableAADFactory;
import net.finmath.montecarlo.automaticdifferentiation.forward.RandomVariableADFactory;
import net.finmath.montecarlo.interestrate.LIBORMarketModel.CalibrationItem;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric.OptimizerDerivativeType;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric.OptimizerSolverType;
import net.finmath.montecarlo.interestrate.modelplugins.DisplacedLocalVolatilityModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCorrelationModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCorrelationModelExponentialDecay;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCovarianceModelFromVolatilityAndCorrelation;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModelPiecewiseConstant;
import net.finmath.montecarlo.interestrate.products.ATMSwaption;
import net.finmath.montecarlo.interestrate.products.AbstractLIBORMonteCarloProduct;
import net.finmath.montecarlo.interestrate.products.Caplet.ValueUnit;
import net.finmath.montecarlo.process.ProcessEulerScheme;
import net.finmath.montecarlo.process.ProcessEulerScheme.Scheme;
import net.finmath.optimizer.OptimizerFactory;
import net.finmath.optimizer.OptimizerFactory.OptimizerType;
import net.finmath.optimizer.OptimizerFactoryInterface;
import net.finmath.optimizer.SolverException;
import net.finmath.stochastic.RandomVariableInterface;
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
	private static DecimalFormat formatterParam		= new DecimalFormat(" #0.000;-#0.000", new DecimalFormatSymbols(Locale.ENGLISH));
	private static DecimalFormat formatterDeviation	= new DecimalFormat(" 0.00000E00;-0.00000E00", new DecimalFormatSymbols(Locale.ENGLISH));

	@Parameters(name="{0}-{2}-{1}")
	public static Collection<Object[]> data() {

		Collection<Object[]> config = new ArrayList<>();
		
		// vector valued calibration
		config.add(new Object[] {OptimizerSolverType.VECTOR, OptimizerDerivativeType.FINITE_DIFFERENCES, OptimizerType.LevenbergMarquardt});
		config.add(new Object[] {OptimizerSolverType.VECTOR, OptimizerDerivativeType.ALGORITHMIC_DIFFERENCIATION, OptimizerType.LevenbergMarquardt});
		config.add(new Object[] {OptimizerSolverType.VECTOR, OptimizerDerivativeType.ADJOINT_ALGORITHMIC_DIFFERENCIATION, OptimizerType.LevenbergMarquardt});
		
		// scalar valued calibration
		config.add(new Object[] {OptimizerSolverType.SKALAR, OptimizerDerivativeType.ADJOINT_ALGORITHMIC_DIFFERENCIATION, OptimizerType.SimpleGradientDescent});
		config.add(new Object[] {OptimizerSolverType.SKALAR, OptimizerDerivativeType.ADJOINT_ALGORITHMIC_DIFFERENCIATION, OptimizerType.GradientDescentArmijo});
		config.add(new Object[] {OptimizerSolverType.SKALAR, OptimizerDerivativeType.ADJOINT_ALGORITHMIC_DIFFERENCIATION, OptimizerType.TruncatedGaussNetwonForUnderdeterminedNSLP});
		config.add(new Object[] {OptimizerSolverType.SKALAR, OptimizerDerivativeType.ADJOINT_ALGORITHMIC_DIFFERENCIATION, OptimizerType.BroydenFletcherGoldfarbShanno});
		
		return config;
	}	

	private final OptimizerSolverType solverType;
	private final OptimizerDerivativeType derivativeType;

	private final OptimizerFactoryInterface optimizerFactory;
	private final AbstractRandomVariableFactory randomVariableFactory;

	private final ValueUnit valueUnit;
	
	public LIBORMarketModelCalibrationTest(OptimizerSolverType solverType, OptimizerDerivativeType derivativeType, OptimizerType optimizerType) {
		// define how derivatives are calculated
		this.solverType = solverType;
		this.derivativeType = derivativeType;
		
		// optimizer settings
		int maxIterations = 100;
		double errorTolerance = 1E-4;		
		
		// 
		this.optimizerFactory = new OptimizerFactory(optimizerType, maxIterations, errorTolerance);
		
		// select for which values the LMM should be calibrated
		this.valueUnit = ValueUnit.VALUE;
		
		System.out.println(solverType + " - " + optimizerType + " - " + derivativeType + "\n");
		
		// define the random variable factory
		Map<String, Object> randomVariableFactoryProperties = new HashMap<>();
		randomVariableFactoryProperties.put("isGradientRetainsLeafNodesOnly", true);

		switch(derivativeType) {
		case ADJOINT_ALGORITHMIC_DIFFERENCIATION:
			this.randomVariableFactory = new RandomVariableDifferentiableAADFactory(new RandomVariableFactory(), randomVariableFactoryProperties);
			break;
		case ALGORITHMIC_DIFFERENCIATION:
			this.randomVariableFactory = new RandomVariableADFactory();
			break;
		default:
			this.randomVariableFactory = new RandomVariableFactory();
		}
	}

	private CalibrationItem createCalibrationItem(double weight, double exerciseDate, double swapPeriodLength, int numberOfPeriods, double moneyness, double targetVolatility, ForwardCurveInterface forwardCurve, DiscountCurveInterface discountCurve, ValueUnit valueUnit) throws CalculationException {

		double[]	fixingDates			= new double[numberOfPeriods];
		double[]	paymentDates		= new double[numberOfPeriods];
		double[]	swapTenor			= new double[numberOfPeriods + 1];

		for (int periodStartIndex = 0; periodStartIndex < numberOfPeriods; periodStartIndex++) {
			fixingDates[periodStartIndex] = exerciseDate + periodStartIndex * swapPeriodLength;
			paymentDates[periodStartIndex] = exerciseDate + (periodStartIndex + 1) * swapPeriodLength;
			swapTenor[periodStartIndex] = exerciseDate + periodStartIndex * swapPeriodLength;
		}
		swapTenor[numberOfPeriods] = exerciseDate + numberOfPeriods * swapPeriodLength;

//		// Swaptions swap rate

//		// Set swap rates for each period
//		double[] swaprates = new double[numberOfPeriods];
//		Arrays.fill(swaprates, swaprate);

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
		case NORMALVOLATILITY:
			calibrationItem = new CalibrationItem(swaptionMonteCarlo, targetVolatility, weight);
			break;
		default:
			throw new UnsupportedOperationException();
		}
		
		return calibrationItem;
	}

	/**
	 * Brute force Monte-Carlo calibration of swaptions.
	 * 
	 * @throws CalculationException
	 * @throws SolverException
	 */
	@Test
	public void testATMSwaptionCalibration() throws CalculationException, SolverException {

		// Simulation Settings
		final int seed				= 1234;
		final int numberOfPaths		= (int) 1E3;
		final int numberOfFactors	= 1;

		//------------------------------------------------------------------------------------------

//		long millisCurvesStart = System.currentTimeMillis();

		/*
		 * Calibration test
		 */
		System.out.println("Calibration to ATM Swaptions.\n");

		/*
		 * Calibration of rate curves
		 */
//		System.out.println("Calibration of rate curves:");

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
		final ArrayList<CalibrationItem>	calibrationItems			= new ArrayList<CalibrationItem>();

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

			calibrationItems.add(createCalibrationItem(weight, exercise, swapPeriodLength, numberOfPeriods, moneyness, targetVolatility, forwardCurve, discountCurve, valueUnit));

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
		//final BrownianMotionInterface brownianMotion = new net.finmath.montecarlo.BrownianMotionCudaWithHostRandomVariable(timeDiscretization, numberOfFactors, numberOfPaths, 31415 /* seed */);
		//final BrownianMotionInterface brownianMotion = new net.finmath.montecarlo.BrownianMotionCudaWithRandomVariableCuda(timeDiscretization, numberOfFactors, numberOfPaths, 31415 /* seed */);

		/* volatility model from piecewise constant interpolated matrix */
		TimeDiscretizationInterface volatilitySurfaceDiscretization = new TimeDiscretization(0.00, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0); 
		double[] initialVolatility = new double[] { LIBORVolatilityModelPiecewiseConstant.parameterTransformInverse(0.50 / 100) };
		LIBORVolatilityModel volatilityModel = new LIBORVolatilityModelPiecewiseConstant(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, volatilitySurfaceDiscretization, volatilitySurfaceDiscretization, initialVolatility, true);

		//		/* volatility model from given matrix */
		//		double initialVolatility = 0.005;
		//		double[][] volatility = new double[timeDiscretization.getNumberOfTimeSteps()][liborPeriodDiscretization.getNumberOfTimeSteps()];
		//		for(int i = 0; i < timeDiscretization.getNumberOfTimeSteps(); i++) Arrays.fill(volatility[i], initialVolatility);
		//		LIBORVolatilityModel volatilityModel = new LIBORVolatilityModelFromGivenMatrix(randomVariableFactory, timeDiscretization, liborPeriodDiscretization, volatility);

		/* Correlation Model with exponential decay */
		LIBORCorrelationModel correlationModel = new LIBORCorrelationModelExponentialDecay(timeDiscretization, liborPeriodDiscretization, numberOfFactors, 0.05, false);

		// Create a covariance model
		//AbstractLIBORCovarianceModelParametric covarianceModelParametric = new LIBORCovarianceModelExponentialForm5Param(timeDiscretization, liborPeriodDiscretization, numberOfFactors, new double[] { 0.20/100.0, 0.05/100.0, 0.10, 0.05/100.0, 0.10} );
		AbstractLIBORCovarianceModelParametric covarianceModelParametric = new LIBORCovarianceModelFromVolatilityAndCorrelation(timeDiscretization, liborPeriodDiscretization, volatilityModel, correlationModel);

		// Create blended local volatility model with fixed parameter (0=lognormal, > 1 = almost a normal model).			
		AbstractLIBORCovarianceModelParametric covarianceModelDisplaced = new DisplacedLocalVolatilityModel(randomVariableFactory, covarianceModelParametric, 1.0/0.25, false /* isCalibrateable */);

		// Set model properties
		Map<String, Object> properties = new HashMap<String, Object>();

		// Choose the simulation measure
		properties.put("measure", LIBORMarketModel.Measure.SPOT.name());

		// Choose normal state space for the Euler scheme (the covariance model above carries a linear local volatility model, such that the resulting model is log-normal).
		properties.put("stateSpace", LIBORMarketModel.StateSpace.NORMAL.name());

		double[] parameterStandardDeviation = new double[covarianceModelParametric.getParameter().length];
		double[] parameterLowerBound = new double[covarianceModelParametric.getParameter().length];
		double[] parameterUpperBound = new double[covarianceModelParametric.getParameter().length];
		Arrays.fill(parameterStandardDeviation, 0.20/100.0);
		Arrays.fill(parameterLowerBound, 0.0);
		Arrays.fill(parameterUpperBound, Double.POSITIVE_INFINITY);


		// Set calibration properties (should use our brownianMotion for calibration - needed to have to right correlation).
		Map<String, Object> calibrationParameters = new HashMap<String, Object>();
		calibrationParameters.put("brownianMotion", brownianMotion);
		calibrationParameters.put("optimizerFactory", optimizerFactory);
		calibrationParameters.put("parameterStep", new Double(1E-4));

		calibrationParameters.put("solverType", solverType);
		calibrationParameters.put("derivativeType", derivativeType);
		calibrationParameters.put("scheme", Scheme.EULER);
		calibrationParameters.put("parameterLowerBound", parameterLowerBound);
		calibrationParameters.put("parameterUpperBound", parameterUpperBound);

		properties.put("calibrationParameters", calibrationParameters);

		long millisCalibrationStart = System.currentTimeMillis();
		/*
		 * Create corresponding LIBOR Market Model
		 */
		LIBORMarketModel.CalibrationItem[] calibrationItemsLMM = new LIBORMarketModel.CalibrationItem[calibrationItemNames.size()];
		for(int i=0; i<calibrationItemNames.size(); i++) calibrationItemsLMM[i] = new LIBORMarketModel.CalibrationItem(calibrationItems.get(i).calibrationProduct,calibrationItems.get(i).calibrationTargetValue,calibrationItems.get(i).calibrationWeight);
		LIBORModelInterface liborMarketModelCalibrated = new LIBORMarketModel(
				liborPeriodDiscretization,
				curveModel,
				forwardCurve, new DiscountCurveFromForwardCurve(forwardCurve),
				randomVariableFactory,
				covarianceModelDisplaced,
				calibrationItemsLMM,
				properties);

		long millisCalibrationEnd = System.currentTimeMillis();

		System.out.println("\nCalibrated parameters are:");
		AbstractLIBORCovarianceModelParametric calibratedCovarianceModel = (AbstractLIBORCovarianceModelParametric) ((LIBORMarketModel) liborMarketModelCalibrated).getCovarianceModel();
		RandomVariableInterface[] param = calibratedCovarianceModel.getParameterAsRandomVariable();
		for (RandomVariableInterface p : param) 
			System.out.println(formatterParam.format(LIBORVolatilityModelPiecewiseConstant.parameterTransform(p).doubleValue()));

		ProcessEulerScheme process = new ProcessEulerScheme(brownianMotion);
		LIBORModelMonteCarloSimulationInterface simulationCalibrated = new LIBORModelMonteCarloSimulation(liborMarketModelCalibrated, process);

		System.out.println("\nValuation on calibrated prices:");
		double deviationSum			= 0.0;
		double deviationSquaredSum	= 0.0;
		for (int i = 0; i < calibrationItems.size(); i++) {
			CalibrationItem calibrationItem = calibrationItems.get(i);
			try {
				double valueModel = calibrationItem.calibrationProduct.getValue(simulationCalibrated);
				double valueTarget = calibrationItem.calibrationTargetValue;
				double error = valueModel-valueTarget;
				deviationSum += error;
				deviationSquaredSum += error*error;
				System.out.println(calibrationItemNames.get(i) + "\t" + "Model: " + formatterValue.format(valueModel) + "\t Target: " + formatterValue.format(valueTarget) + "\t Deviation: " + formatterDeviation.format(valueModel-valueTarget));// + "\t" + calibrationProduct.toString());
			}
			catch(Exception e) {
			}
		}

		//		System.out.println("Calibration of curves........." + (millisCurvesEnd-millisCurvesStart)/1000.0);
		System.out.println("Calibration of volatilities..." + (millisCalibrationEnd-millisCalibrationStart)/1000.0);
		System.out.println("Number of Iterations.........." + covarianceModelDisplaced.getCalibrationOptimizer().getIterations());

		double averageDeviation = deviationSum/calibrationItems.size();
		System.out.println("Mean Deviation:" + formatterValue.format(averageDeviation));
		System.out.println("RMS Error.....:" + formatterValue.format(Math.sqrt(deviationSquaredSum/calibrationItems.size())));
		System.out.println("__________________________________________________________________________________________\n");

		// evaluate the two deviation averages
		Assert.assertEquals(0.0, averageDeviation, 1E-4);
	}

	public AnalyticModelInterface getCalibratedCurve() throws SolverException {
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

		// Get best parameters
		//		double[] parametersBest = calibratedModel.getDiscountCurve(discountCurve.getName()).getParameter();

		// Test calibration
		model			= calibratedModel;

//		double squaredErrorSum = 0.0;
//		for(AnalyticProductInterface c : calibrationProducts) {
//			double value = c.getValue(0.0, model);
//			double valueTaget = 0.0;
//			double error = value - valueTaget;
//			squaredErrorSum += error*error;
//		}
//		double rms = Math.sqrt(squaredErrorSum/calibrationProducts.size());

//		System.out.println("Independent checked acccurary: " + rms);

//		System.out.println("Calibrated discount curve: ");
//		System.out.println("Maturity" + "\t" + "Discount Values");
//		for(int i=0; i<curveMaturities.length; i++) {
//			double maturity = curveMaturities[i];
//			double discountValue = calibratedModel.getDiscountCurve(discountCurve.getName()).getDiscountFactor(maturity);
//			System.out.println(formatterParam.format(maturity) + "\t\t" + formatterValue.format(discountValue));
//		}
		return model;
	}

	private static double getParSwaprate(ForwardCurveInterface forwardCurve, DiscountCurveInterface discountCurve, double[] swapTenor) throws CalculationException {
		return net.finmath.marketdata.products.Swap.getForwardSwapRate(new TimeDiscretization(swapTenor), new TimeDiscretization(swapTenor), forwardCurve, discountCurve);
	}
}