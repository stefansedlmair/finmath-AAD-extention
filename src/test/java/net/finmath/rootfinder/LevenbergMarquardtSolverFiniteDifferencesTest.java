package net.finmath.rootfinder;

import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;

import org.junit.Assert;
import org.junit.Test;

import net.finmath.exception.CalculationException;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurveInterface;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.IndependentIncrementsInterface;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.montecarlo.automaticdifferentiation.backward.alternative.RandomVariableAADv3Factory;
import net.finmath.montecarlo.process.AbstractProcess;
import net.finmath.montecarlo.process.ProcessEulerScheme;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretization.ShortPeriodLocation;
import net.finmath.time.TimeDiscretizationInterface;

public class LevenbergMarquardtSolverFiniteDifferencesTest {

	AbstractRandomVariableFactory randomVariableFactory = new RandomVariableAADv3Factory();
	
	@Test
	public void ProcessEulerSchemeFDTest() throws CalculationException{
		
		/* set up Problem */
		
		double swaprate 				= -0.01;
		
		double startTime 				= 0.0;
		double endTime 					= 10.0;
		
		double evaluationTime 			= 1.0;
		double exerciseDate 			= 7.0;
		
		int numberOfLibors 				= 20;
		int numberOfSiumlationSteps 	= 20;
		int numberOfSwapPayments 		= 3;
		
		int seed 						= 1234;
		int numberOfFactors 			= 4; 
		int numberOfPaths 				= (int) Math.pow(10, 2);
		double h 						= 		Math.pow(10, -6);
		
		TimeDiscretizationInterface liborPeriodDiscretization 	= new TimeDiscretization(startTime, endTime, (endTime - startTime)/numberOfLibors, ShortPeriodLocation.SHORT_PERIOD_AT_END);
		TimeDiscretizationInterface swapTenor 					= new TimeDiscretization(startTime, endTime, (endTime - startTime)/numberOfSwapPayments, ShortPeriodLocation.SHORT_PERIOD_AT_END);
		TimeDiscretizationInterface simulationTenorStructur 	= new TimeDiscretization(startTime, endTime, (endTime - startTime)/numberOfSiumlationSteps, ShortPeriodLocation.SHORT_PERIOD_AT_END);
		
		IndependentIncrementsInterface brownianMotion 			= new BrownianMotion(simulationTenorStructur, numberOfFactors, numberOfPaths, seed, randomVariableFactory);
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
		
		RandomVariableInterface targetSwapPrice = LevenbergMarquardtSolverTest.swaptionPricer(simulationTenorStructur, liborPeriodDiscretization, swapTenor, forwardRateCurve, eulerScheme, numberOfFactors, exerciseDate, evaluationTime, swaprate, targetCovarianceParameters);
	
		/* set up solver */
		
		int maxNumberOfIterations = 30;
		double targetAccuracy = Math.pow(10, -9);
		
		
		LevenbergMarquardtSolver LMSolver = new LevenbergMarquardtSolver(LevenbergMarquardtSolverTest.ParameterArrayToTreeMap(initialCovarianceParameters), targetSwapPrice, targetAccuracy, maxNumberOfIterations);
		
		
		long totalTime = 0;
		long totalMem = 0;
		
		while(!LMSolver.isDone()){
			
			long startMem = LevenbergMarquardtSolverTest.getAllocatedMemory();
			long startClock = System.currentTimeMillis();
			
			Map<Long, RandomVariableInterface> nextCovarianceParameterMap = LMSolver.getNextParameters();
			
			RandomVariableInterface[] currentCovarianceParameters = LevenbergMarquardtSolverTest.ParameterMapToDifferentiableArray(nextCovarianceParameterMap);
			
			RandomVariableInterface currentSwapPrice = LevenbergMarquardtSolverTest.swaptionPricer(simulationTenorStructur, liborPeriodDiscretization, swapTenor, forwardRateCurve, eulerScheme, 
					numberOfFactors, exerciseDate, evaluationTime, swaprate, currentCovarianceParameters);
			
			// FD
			Map<Long, RandomVariableInterface> swaptionSensitivitiesFD = new TreeMap<>();
			
			for(int parameterIndex = 0; parameterIndex < currentCovarianceParameters.length; parameterIndex++){
				RandomVariableInterface[] covarianceParametersPlus = currentCovarianceParameters.clone();
				covarianceParametersPlus[parameterIndex] = covarianceParametersPlus[parameterIndex].add(h); 
				RandomVariableInterface[] covarianceParametersMinus = currentCovarianceParameters.clone();
				covarianceParametersMinus[parameterIndex] = covarianceParametersPlus[parameterIndex].sub(h);
				
				RandomVariableInterface swaptionPricePlus = LevenbergMarquardtSolverTest.swaptionPricer(simulationTenorStructur, liborPeriodDiscretization, swapTenor, forwardRateCurve, eulerScheme, numberOfFactors, exerciseDate, evaluationTime, swaprate, covarianceParametersPlus);
				RandomVariableInterface swaptionPriceMinus = LevenbergMarquardtSolverTest.swaptionPricer(simulationTenorStructur, liborPeriodDiscretization, swapTenor, forwardRateCurve, eulerScheme, numberOfFactors, exerciseDate, evaluationTime, swaprate, covarianceParametersMinus);

				swaptionSensitivitiesFD.put(((RandomVariableDifferentiableInterface) currentCovarianceParameters[parameterIndex]).getID(), 
						swaptionPricePlus.sub(swaptionPriceMinus).div(2*h));
			}		
			
			LMSolver.setValueAndDerivative(currentSwapPrice, swaptionSensitivitiesFD);
			
			long stopClock = System.currentTimeMillis();
			long endMem = LevenbergMarquardtSolverTest.getAllocatedMemory();
			
			totalTime += stopClock - startClock;
			totalMem += endMem - startMem;
			
//			System.out.println("Step#:..............." + LMSolver.getNumberOfIterations());
//			System.out.println("Accuracy:............" + LMSolver.getAccuracy());
//			System.out.println("Lambda:.............." + LMSolver.getLambda());
//			System.out.println("duration:............" + (double)(stopClock - startClock)/1000.0 + "s");
//			System.out.println("memory consumption:.." + (double)(endMem - startMem)/Math.pow(10, 6) + "Mbyte");
		}
		
		System.out.println("number of Iterations........................." + LMSolver.getNumberOfIterations());
		System.out.println("average duration per Iteration..............." + (double)totalTime/ ((double)LMSolver.getNumberOfIterations() *1000.0) +"s ");
		System.out.println("average memory consumption per Iteration....." + (double)totalMem/ ((double)LMSolver.getNumberOfIterations() * Math.pow(10, 6)) +"Mbyte");
		System.out.println();
		System.out.println("best accuracy................................" + LMSolver.getAccuracy());
		
		double[] finalParameters = new double[5];
		int i = 0;
		for(Long key: LMSolver.getBestParameters().keySet())
			finalParameters[i++] = LMSolver.getBestParameters().get(key).getAverage();
		
		System.out.println("best parameters.............................." + Arrays.toString(finalParameters));		
		System.out.println();
		Assert.assertTrue(LMSolver.getAccuracy() < targetAccuracy );

	}
}
