/**
 * 
 */
package net.finmath.montecarlo.interestrate;

import java.util.HashMap;
import java.util.Map;

import net.finmath.exception.CalculationException;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric.OptimizerDerivativeType;
import net.finmath.montecarlo.interestrate.modelplugins.AbstractLIBORCovarianceModelParametric.OptimizerSolverType;
import net.finmath.montecarlo.interestrate.products.SwaptionSimple.ValueUnit;
import net.finmath.optimizer.OptimizerFactory.OptimizerType;
import net.finmath.optimizer.SolverException;

/**
 * @author Stefan Sedlmair
 *
 */
public class LIBORMarketModelCalibrationExecutableATM {

	/**
	 * @param args
	 * @throws CalculationException 
	 * @throws SolverException 
	 */
	public static void main(String[] args) throws CalculationException, SolverException {
		
		Map<String, Object> testProperties = new HashMap<>();
	
		// args[0] = value unit
		final ValueUnit valueUnit = ValueUnit.valueOf(args[0]);
		testProperties.put("ValueUnit", valueUnit);
	
		// args[1] = numberOfPaths
		final int numberOfPaths = Integer.valueOf(args[1]);
		testProperties.put("numberOfPaths", numberOfPaths);

		// args[2] = solverType
		final OptimizerSolverType solverType = OptimizerSolverType.valueOf(args[2]);
		testProperties.put("SolverType", solverType);

		
		// args[3] = derivativeType
		final OptimizerDerivativeType derivativeType = OptimizerDerivativeType.valueOf(args[3]);
		testProperties.put("DerivativeType", derivativeType);

		// args[4]
		final OptimizerType optimizerType = OptimizerType.valueOf(args[4]);
		testProperties.put("OptimizerType", optimizerType);

		// args[5] = maxIterations
		final int maxIterations = Integer.valueOf(args[5]);
		testProperties.put("maxIterations", maxIterations);

		// args[6] = errorTolerance	
		final double errorTolerance = Double.valueOf(args[6]);		
		testProperties.put("errorTolerance", errorTolerance);

		// args[7] = errorTolerance	
		final int numberOfThreads = Integer.valueOf(args[7]);		
		testProperties.put("numberOfThreads", numberOfThreads);

		// args[8] = seed
		final int seed = Integer.valueOf(args[8]);
		testProperties.put("seed", seed);

		// args[9] = numberOfFactors
		final int numberOfFactors = Integer.valueOf(args[9]);
		testProperties.put("numberOfFactors", numberOfFactors);

		LIBORMarketModelCalibrationTest.ATMSwaptionCalibration(testProperties);
	}	
}
