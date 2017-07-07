/**
 * 
 */
package net.finmath.rootfinder;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.TreeMap;

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
import net.finmath.montecarlo.interestrate.modelplugins.LIBORCovarianceModelExponentialForm5Param;
import net.finmath.montecarlo.interestrate.products.Swaption;
import net.finmath.montecarlo.process.AbstractProcess;
import net.finmath.montecarlo.process.DifferentiableProcessEulerScheme;
import net.finmath.montecarlo.process.ProcessEulerScheme;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretization.ShortPeriodLocation;
import net.finmath.time.TimeDiscretizationInterface;


/**
 * @author Stefan Sedlmair
 *
 */
@RunWith(Parameterized.class)
public class LIBORMarktModelSensitivityTest {

	@Parameters
    public static Collection<Object[]> data(){
        return Arrays.asList(new Object[][] {
        	{new RandomVariableDifferentiableAAD2Factory()},
        	{new RandomVariableAADv3Factory()}
        });
    }

    private final AbstractRandomVariableFactory randomVariableFactory;
    private final AbstractRandomVariableFactory nonfdiffereniableRandomVariableFactory;
    
    public LIBORMarktModelSensitivityTest(AbstractRandomVariableFactory factory) {
    	
    	this.randomVariableFactory = factory;
    	this.nonfdiffereniableRandomVariableFactory = new RandomVariableFactory();
    	
    	System.out.print(randomVariableFactory.getClass().getSimpleName() + " - ");
    }
    
	@Test
    public void testSwaptionSensitivities() throws CalculationException {

		System.out.println(" Swaption Sensitivities");
		
		double swaprate 				= -0.1;
		
		double startTime 				= 0.0;
		double endTime 					= 10.0;
		
		double evaluationTime 			= 1.0;
		double exerciseDate 			= 7.0;
		
		int numberOfLibors 				= 20;
		int numberOfSiumlationSteps 	= 20;
		int numberOfSwapPayments 		= 10;
		
		int seed 						= 1234;
		int numberOfFactors 			= 2; 
		int numberOfPaths 				= (int) Math.pow(10, 2);
		double h 						= 		Math.pow(10, -6);
		
		TimeDiscretizationInterface liborPeriodDiscretization 	= new TimeDiscretization(startTime, endTime, (endTime - startTime)/numberOfLibors, ShortPeriodLocation.SHORT_PERIOD_AT_END);
		TimeDiscretizationInterface swapTenor 					= new TimeDiscretization(startTime, endTime, (endTime - startTime)/numberOfSwapPayments, ShortPeriodLocation.SHORT_PERIOD_AT_END);
		TimeDiscretizationInterface simulationTenorStructur 	= new TimeDiscretization(startTime, endTime, (endTime - startTime)/numberOfSiumlationSteps, ShortPeriodLocation.SHORT_PERIOD_AT_END);
		
		IndependentIncrementsInterface brownianMotion 			= new BrownianMotion(simulationTenorStructur, numberOfFactors, numberOfPaths, seed, nonfdiffereniableRandomVariableFactory);
		AbstractProcess eulerScheme 							= new ProcessEulerScheme(brownianMotion);
		
		ForwardCurveInterface forwardRateCurve = ForwardCurve.createForwardCurveFromForwards("ForwardCurve", 
				new double[]{0.0, 	1.0, 	2.0, 	3.0, 	10.0, 	20.0, 	30.0} /* times */, 
				new double[]{0.0, 	0.01, 	0.02, 	0.02, 	0.03, 	0.04, 	0.05} /* givenForwards */, 
				0.01 /* paymentOffset */);
		
		RandomVariableInterface[] covarianceParameters 	= new RandomVariableInterface[]{
				randomVariableFactory.createRandomVariable(0.1), 	/* a */
				randomVariableFactory.createRandomVariable(0.2), 	/* b */
				randomVariableFactory.createRandomVariable(0.3), 	/* c */
				randomVariableFactory.createRandomVariable(0.4), 	/* d */
				randomVariableFactory.createRandomVariable(0.5)  	/* correlationParameter */
		};
		
		RandomVariableInterface swaptionValue  = getSwaptionPrice(liborPeriodDiscretization, simulationTenorStructur, forwardRateCurve, swapTenor, exerciseDate, evaluationTime, swaprate, eulerScheme, numberOfFactors, covarianceParameters);
		
		// AAD
		Map<Long, RandomVariableInterface> swaptionSensitivitiesAAD = ((RandomVariableDifferentiableInterface) swaptionValue).getGradient();
	
		// FD
		Map<Long, RandomVariableInterface> swaptionSensitivitiesFD = new TreeMap<>();
		for(int parameterIndex = 0; parameterIndex < covarianceParameters.length; parameterIndex++){
			RandomVariableInterface[] covarianceParametersPlus = covarianceParameters.clone();
			covarianceParametersPlus[parameterIndex] = covarianceParametersPlus[parameterIndex].add(h); 
			RandomVariableInterface[] covarianceParametersMinus = covarianceParameters.clone();
			covarianceParametersMinus[parameterIndex] = covarianceParametersPlus[parameterIndex].sub(h);
			
			RandomVariableInterface swaptionPricePlus = getSwaptionPrice(liborPeriodDiscretization, simulationTenorStructur, forwardRateCurve, swapTenor, exerciseDate, evaluationTime, swaprate, eulerScheme, numberOfFactors, covarianceParametersPlus);
			RandomVariableInterface swaptionPriceMinus = getSwaptionPrice(liborPeriodDiscretization, simulationTenorStructur, forwardRateCurve, swapTenor, exerciseDate, evaluationTime, swaprate, eulerScheme, numberOfFactors, covarianceParametersMinus);

			swaptionSensitivitiesFD.put(((RandomVariableDifferentiableInterface) covarianceParameters[parameterIndex]).getID(), 
					swaptionPricePlus.sub(swaptionPriceMinus).div(2*h));
		}
		
		
		for(Long key : swaptionSensitivitiesAAD.keySet()){
			System.out.println();
			System.out.println("ParameterID:.............." + key);
			System.out.println("derivative FD:............" + Arrays.toString(swaptionSensitivitiesFD.get(key).getRealizations()));
			System.out.println("derivative AAD:..........." + Arrays.toString(swaptionSensitivitiesAAD.get(key).getRealizations()));
		}
		System.out.println();
	}
	
	private RandomVariableInterface getSwaptionPrice(
			TimeDiscretizationInterface liborPeriodDiscretization,
			TimeDiscretizationInterface simulationTenorStructur,
			ForwardCurveInterface forwardRateCurve, 
			TimeDiscretizationInterface swapTenor, 
			double exerciseDate, 
			double evaluationTime, 
			double swaprate,  
			AbstractProcess eulerScheme, 
			int numberOfFactors, 
			RandomVariableInterface[] covarianceParameters) throws CalculationException{
		AbstractLIBORCovarianceModel liborCovarianceModel = new LIBORCovarianceModelExponentialForm5Param(simulationTenorStructur, liborPeriodDiscretization, numberOfFactors, covarianceParameters);
		LIBORModelInterface liborModel = new LIBORMarketModel(liborPeriodDiscretization, forwardRateCurve, liborCovarianceModel);
		
		LIBORModelMonteCarloSimulationInterface liborMCmodel = new LIBORModelMonteCarloSimulation(liborModel, eulerScheme.clone());
		
		Swaption swaption = new Swaption(exerciseDate, swapTenor, swaprate);
		return swaption.getValue(evaluationTime, liborMCmodel);
	}

}
