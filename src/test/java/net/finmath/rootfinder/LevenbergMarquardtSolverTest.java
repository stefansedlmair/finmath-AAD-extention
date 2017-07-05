/**
 * 
 */
package net.finmath.rootfinder;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.TreeMap;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import net.finmath.exception.CalculationException;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurveInterface;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.IndependentIncrementsInterface;
import net.finmath.montecarlo.RandomVariable;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.montecarlo.automaticdifferentiation.backward.RandomVariableDifferentiableAAD2Factory;
import net.finmath.montecarlo.automaticdifferentiation.backward.RandomVariableDifferentiableAADFactory;
import net.finmath.montecarlo.automaticdifferentiation.backward.alternative.RandomVariableAADv2Factory;
import net.finmath.montecarlo.automaticdifferentiation.backward.alternative.RandomVariableAADv3Factory;
import net.finmath.montecarlo.interestrate.LIBORMarketModel;
import net.finmath.montecarlo.interestrate.LIBORModelInterface;
import net.finmath.montecarlo.interestrate.LIBORModelMonteCarloSimulation;
import net.finmath.montecarlo.interestrate.LIBORModelMonteCarloSimulationInterface;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModel;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCorrelationModelExponentialDecay;
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCovarianceModelExponentialForm5Param;
import net.finmath.montecarlo.interestrate.products.Swaption;
import net.finmath.montecarlo.process.AbstractProcess;
import net.finmath.montecarlo.process.ProcessEulerScheme;
import net.finmath.randomnumbers.MersenneTwister;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationInterface;
import net.finmath.time.TimeDiscretization.ShortPeriodLocation;



/**
 * @author Stefan Sedlmair
 *
 */
@RunWith(Parameterized.class)
public class LevenbergMarquardtSolverTest {
	
	/* parameters specify the factories one wants to test against each other */
	@Parameters
    public static Collection<Object[]> data(){
        return Arrays.asList(new Object[][] {
        	{new RandomVariableDifferentiableAAD2Factory()},
//        	{new RandomVariableDifferentiableAADFactory()},
//        	{new RandomVariableAADv3Factory()},
        	{new RandomVariableAADv3Factory()}
        });
    }

    private final AbstractRandomVariableFactory randomVariableFactory;
    private final AbstractRandomVariableFactory nonfdiffereniableRandomVariableFactory;
    
    public LevenbergMarquardtSolverTest(AbstractRandomVariableFactory factory) {
    	
    	this.randomVariableFactory = factory;
    	this.nonfdiffereniableRandomVariableFactory = new RandomVariableFactory();
    	
    	System.out.println(randomVariableFactory.getClass().getSimpleName());
    }
    
	@Test
	public void PolynomialFittingTest() {
	
		/* --------------------- problem set up -------------------------*/
		
		long seed  = 1234;
		MersenneTwister RNG = new MersenneTwister(seed);
		
		int numberOfRealization = (int)Math.pow(10, 5);
		int numberOfArguments 	= 10;
		
		// create random vectors for arguments
		TreeMap<Long, RandomVariableInterface> initialArguments = new TreeMap<>();
		
		double[] realizations = new double[numberOfRealization];
		
		// generate x argument
		for(int j = 0; j < numberOfRealization; j++)
			realizations[j] = RNG.nextDouble();
		RandomVariableInterface x = new RandomVariable(0.0, realizations);
		
		// generate target function value
		RandomVariableInterface targetFunctionValue = new RandomVariable(0.0, RNG.nextDouble());
		
		// generate arguments that shall be fitted
		for(int i = 0; i < numberOfArguments; i++){
			
			RandomVariableDifferentiableInterface argument = (RandomVariableDifferentiableInterface) randomVariableFactory.createRandomVariable(0.0, RNG.nextDouble());
			
			initialArguments.put(argument.getID(), argument);
		}
		
		/* --------------------- solver set up -------------------------*/
		
		
		int maxNumberOfIterations = 30;
		double targetAccuracy = Math.pow(10, -6);
		
		
		LevenbergMarquardtSolver LMSolver = new LevenbergMarquardtSolver(initialArguments, targetFunctionValue, targetAccuracy, maxNumberOfIterations);

		RandomVariableDifferentiableInterface[] nextParametersArray = new RandomVariableDifferentiableInterface[numberOfArguments];

		long totalTime = 0;
		long totalMem = 0;
		
		while(!LMSolver.isDone()){
			
			long startMem = getAllocatedMemory();
			long startTime = System.currentTimeMillis();
			
			RandomVariableDifferentiableInterface currentFunctionValue = polynomialFuction(x, ParameterMapToDifferentiableArray(LMSolver.getNextParameters()));
			
			LMSolver.setValueAndDerivative(currentFunctionValue, currentFunctionValue.getGradient());
			
			long endTime = System.currentTimeMillis();
			long endMem = getAllocatedMemory();
			
//			System.out.println("Step#:..............." + LMSolver.getNumberOfIterations());
//			System.out.println("Accuracy:............" + LMSolver.getAccuracy());
//			System.out.println("Lambda:.............." + LMSolver.getLambda());
//			System.out.println("duration:............" + (double)(endTime - startTime)/1000.0 + "s");
//			System.out.println("memory consumption:.." + (double)(endMem - startMem)/Math.pow(10, 6) + "Mbyte");

			totalTime += endTime - startTime;
			totalMem += endMem - startMem;
		}
		System.out.println("number of Iterations........................." + LMSolver.getNumberOfIterations());
		System.out.println("average duration per Iteration..............." + (double)totalTime/ ((double)LMSolver.getNumberOfIterations() *1000.0) +"s ");
		System.out.println("average memory consumption per Iteration....." + (double)totalMem/ ((double)LMSolver.getNumberOfIterations() * Math.pow(10, 6)) +"Mbyte");
		System.out.println();
		System.out.println("best accuracy................................" + LMSolver.getAccuracy());
//		System.out.println("best parameters.............................." + LMSolver.getBestParameters());		
		System.out.println();
		Assert.assertTrue(LMSolver.getAccuracy() < targetAccuracy );
			
	}
	
	private RandomVariableDifferentiableInterface polynomialFuction(RandomVariableInterface x, RandomVariableDifferentiableInterface[] a){
		RandomVariableInterface polynom = a[0];
		for(int i = 1; i < a.length; i++)
			polynom = polynom.addProduct(a[i], x.pow(i));
			
		return (RandomVariableDifferentiableInterface) polynom.average();
	}
	
	@Test
	public void ProcessEulerSchemeTest() throws CalculationException{
		
		/* set up Problem */
		
		double swaprate 				= 0.001;
		
		double startTime 				= 0.0;
		double endTime 					= 10.0;
		
		double evaluationTime 			= 1.0;
		double exerciseDate 			= 7.0;
		
		int numberOfLibors 				= 20;
		int numberOfSiumlationSteps 	= 20;
		int numberOfSwapPayments 		= 3;
		
		int seed 						= 1234;
		int numberOfFactors 			= 2; 
		int numberOfPaths 				= (int) Math.pow(10, 4);
		
		TimeDiscretizationInterface liborPeriodDiscretization 	= new TimeDiscretization(startTime, endTime, (endTime - startTime)/numberOfLibors, ShortPeriodLocation.SHORT_PERIOD_AT_END);
		TimeDiscretizationInterface swapTenor 					= new TimeDiscretization(startTime, endTime, (endTime - startTime)/numberOfSwapPayments, ShortPeriodLocation.SHORT_PERIOD_AT_END);
		TimeDiscretizationInterface simulationTenorStructur 	= new TimeDiscretization(startTime, endTime, (endTime - startTime)/numberOfSiumlationSteps, ShortPeriodLocation.SHORT_PERIOD_AT_END);
		
		IndependentIncrementsInterface brownianMotion 			= new BrownianMotion(simulationTenorStructur, numberOfFactors, numberOfPaths, seed, nonfdiffereniableRandomVariableFactory);
		AbstractProcess eulerScheme 							= new ProcessEulerScheme(brownianMotion);
		
		ForwardCurveInterface forwardRateCurve = ForwardCurve.createForwardCurveFromForwards("ForwardCurve", 
				new double[]{0.0, 	1.0, 	2.0, 	3.0, 	10.0, 	20.0, 	30.0} /* times */, 
				new double[]{0.0, 	0.01, 	0.02, 	0.02, 	0.03, 	0.04, 	0.05} /* givenForwards */, 
				0.01 /* paymentOffset */);
		
		RandomVariableInterface[] initialCovarianceParameters 	= new RandomVariableInterface[]{
				randomVariableFactory.createRandomVariable(0.1), 	/* a */
				randomVariableFactory.createRandomVariable(0.1), 	/* b */
				randomVariableFactory.createRandomVariable(0.1), 	/* c */
				randomVariableFactory.createRandomVariable(0.1), 	/* d */
				randomVariableFactory.createRandomVariable(0.1)  	/* correlationParameter */
		};
		
		RandomVariableInterface[] targetCovarianceParameters 	= new RandomVariableInterface[]{
				randomVariableFactory.createRandomVariable(0.3), 	/* a */
				randomVariableFactory.createRandomVariable(0.1), 	/* b */
				randomVariableFactory.createRandomVariable(0.2), 	/* c */
				randomVariableFactory.createRandomVariable(0.4), 	/* d */
				randomVariableFactory.createRandomVariable(0.5)  	/* correlationParameter */
		};
		
		RandomVariableInterface targetSwapPrice = swaptionPricer(simulationTenorStructur, liborPeriodDiscretization, swapTenor, forwardRateCurve, eulerScheme, numberOfFactors, exerciseDate, evaluationTime, swaprate, targetCovarianceParameters);
	
		/* set up solver */
		
		int maxNumberOfIterations = 30;
		double targetAccuracy = Math.pow(10, -6);
		
		
		LevenbergMarquardtSolver LMSolver = new LevenbergMarquardtSolver(ParameterArrayToTreeMap(initialCovarianceParameters), targetSwapPrice, targetAccuracy, maxNumberOfIterations);
		
		
		long totalTime = 0;
		long totalMem = 0;
		
		while(!LMSolver.isDone()){
			
			long startMem = getAllocatedMemory();
			long startClock = System.currentTimeMillis();
			
			Map<Long, RandomVariableInterface> nextCovarianceParameterMap = LMSolver.getNextParameters();
			
			RandomVariableInterface currentSwapPrice = swaptionPricer(simulationTenorStructur, liborPeriodDiscretization, swapTenor, forwardRateCurve, eulerScheme, 
					numberOfFactors, exerciseDate, evaluationTime, swaprate, ParameterMapToDifferentiableArray(nextCovarianceParameterMap));
			
			Map<Long, RandomVariableInterface> gradient = ((RandomVariableDifferentiableInterface) currentSwapPrice).getGradient();
			
			LMSolver.setValueAndDerivative(currentSwapPrice, gradient);
			
			long stopClock = System.currentTimeMillis();
			long endMem = getAllocatedMemory();
			
			totalTime += stopClock - startClock;
			totalMem += endMem - startMem;
			
			System.out.println("Step#:..............." + LMSolver.getNumberOfIterations());
			System.out.println("Accuracy:............" + LMSolver.getAccuracy());
			System.out.println("Lambda:.............." + LMSolver.getLambda());
			System.out.println("duration:............" + (double)(stopClock - startClock)/1000.0 + "s");
			System.out.println("memory consumption:.." + (double)(endMem - startMem)/Math.pow(10, 6) + "Mbyte");
		}
		
		System.out.println("number of Iterations........................." + LMSolver.getNumberOfIterations());
		System.out.println("average duration per Iteration..............." + (double)totalTime/ ((double)LMSolver.getNumberOfIterations() *1000.0) +"s ");
		System.out.println("average memory consumption per Iteration....." + (double)totalMem/ ((double)LMSolver.getNumberOfIterations() * Math.pow(10, 6)) +"Mbyte");
		System.out.println();
		System.out.println("best accuracy................................" + LMSolver.getAccuracy());
//		System.out.println("best parameters.............................." + LMSolver.getBestParameters());		
		System.out.println();
		Assert.assertTrue(LMSolver.getAccuracy() < targetAccuracy );

	}
	
	private RandomVariableInterface swaptionPricer(			
			TimeDiscretizationInterface simulationTenorStructur, 
			TimeDiscretizationInterface liborPeriodDiscretization,
			TimeDiscretizationInterface swapTenor,
			ForwardCurveInterface forwardRateCurve,
			AbstractProcess eulerScheme,
			int numberOfFactors,
			double exerciseDate,
			double evaluationTime,
			double swaprate,
			RandomVariableInterface[] covarianceParameters) throws CalculationException{
			
		AbstractLIBORCovarianceModel liborCovarianceModel = new LIBORCovarianceModelExponentialForm5Param(simulationTenorStructur, liborPeriodDiscretization, numberOfFactors, covarianceParameters);
		LIBORModelInterface liborModel = new LIBORMarketModel(liborPeriodDiscretization, forwardRateCurve, liborCovarianceModel);
		
		LIBORModelMonteCarloSimulationInterface liborMCmodel = new LIBORModelMonteCarloSimulation(liborModel, eulerScheme.clone());
		
		Swaption swaption = new Swaption(exerciseDate, swapTenor, swaprate);
		RandomVariableInterface swaptionValue = swaption.getValue(evaluationTime, liborMCmodel);
		
		return swaptionValue.average();
	}
	
	
	static long getAllocatedMemory() {
		System.gc();
		long allocatedMemory = (Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory());
		return allocatedMemory;
	}
	
	static TreeMap<Long, RandomVariableInterface> ParameterArrayToTreeMap(RandomVariableInterface[] parameters){
		TreeMap<Long, RandomVariableInterface> parameterMap = new TreeMap<>();
		
		for(RandomVariableInterface parameter : parameters)
			parameterMap.put(((RandomVariableDifferentiableInterface) parameter).getID(), parameter);
		
		return parameterMap;
	}
	
	static RandomVariableDifferentiableInterface[] ParameterMapToDifferentiableArray(Map<Long, RandomVariableInterface> parameterMap){
		RandomVariableDifferentiableInterface[] differentiableParameters = new RandomVariableDifferentiableInterface[parameterMap.size()];
		
		int argumentIndex = 0;
		for(long key : parameterMap.keySet())
			differentiableParameters[argumentIndex++] = (RandomVariableDifferentiableInterface) parameterMap.get(key);
		
		return differentiableParameters;
	}

}
