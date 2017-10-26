/**
 * 
 */
package net.finmath.optimizer;

import java.util.HashMap;
import java.util.Map;

import net.finmath.optimizer.OptimizerInterface.ObjectiveFunction;
import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;
import net.finmath.optimizer.gradientdescent.GradientDescentArmijosRule;
import net.finmath.optimizer.gradientdescent.SimpleGradientDescent;
import net.finmath.optimizer.gradientdescent.TruncatedGaussNewtonForUnterdetNSLP;
import net.finmath.optimizer.quasinewton.BroydenFletcherGoldfarbShanno;

/**
 * General Optimizer Factory able to return any optimizer implemented for calibrating an LMM
 * 
 * @author Stefan Sedlmair
 */
public class OptimizerFactory implements OptimizerFactoryInterface {

	public static enum OptimizerType {
		LevenbergMarquardt, SimpleGradientDescent, GradientDescentArmijo, TruncatedGaussNetwonForUnderdeterminedNSLP, BroydenFletcherGoldfarbShanno
	}

	private final int		maxIterations;
	private final double	errorTolerance;
	private final int		maxThreads;

	private final OptimizerType 		optimizerType;
	private final Map<String, Object> 	properties; 

	public OptimizerFactory(OptimizerType optimizerType, int maxIterations, double errorTolerance, int maxThreads, Map<String, Object> properties) {
		super();
		this.maxIterations 	= maxIterations;
		this.errorTolerance = errorTolerance;
		this.maxThreads 	= maxThreads;

		this.optimizerType 	= optimizerType;
		this.properties 	= properties;
	}

	public OptimizerFactory(OptimizerType optimizerType, int maxIterations, double errorTolerance) {
		this(optimizerType, maxIterations, errorTolerance, 2 /*maxThreads*/, new HashMap<>());
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterface.ObjectiveFunction, double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters,	double[] targetValues) {

		OptimizerInterface optimizer = null;

		switch (optimizerType) {
		case BroydenFletcherGoldfarbShanno:
			if(targetValues.length > 1) throw new IllegalArgumentException();
			optimizer = new BroydenFletcherGoldfarbShanno(initialParameters, targetValues[0], maxIterations, errorTolerance) {

				@Override
				public void setValues(double[] parameters, double[] values) throws SolverException {
					objectiveFunction.setValues(parameters, values);
				}
			};
			break;
		case GradientDescentArmijo:
			if(targetValues.length > 1) throw new IllegalArgumentException();
			optimizer = new GradientDescentArmijosRule(initialParameters, targetValues[0], errorTolerance, maxIterations) {

				@Override
				public void setValues(double[] parameters, double[] values) throws SolverException {
					objectiveFunction.setValues(parameters, values);
				}
			};
			break;
		case SimpleGradientDescent:
			if(targetValues.length > 1) throw new IllegalArgumentException();
			optimizer = new SimpleGradientDescent(initialParameters, targetValues[0], errorTolerance, maxIterations) {

				@Override
				public void setValues(double[] parameters, double[] values) throws SolverException {
					objectiveFunction.setValues(parameters, values);
				}
			};
			break;
		case TruncatedGaussNetwonForUnderdeterminedNSLP:
			if(targetValues.length > 1) throw new IllegalArgumentException();
			optimizer = new TruncatedGaussNewtonForUnterdetNSLP(initialParameters, targetValues[0], errorTolerance, maxIterations) {

				@Override
				public void setValues(double[] parameters, double[] values) throws SolverException {
					objectiveFunction.setValues(parameters, values);
				}
			};
			break;
		case LevenbergMarquardt:
			optimizer = (new LevenbergMarquardt(initialParameters, targetValues, maxIterations, maxThreads) {

				@Override
				public void setValues(double[] parameters, double[] values) throws SolverException {
					objectiveFunction.setValues(parameters, values);
				}
			}).setErrorTolerance(errorTolerance);
			break;
		default:
			throw new UnknownError();		
		}

		return optimizer;
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterface.ObjectiveFunction, double[], double[], double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters,	double[] lowerBound, double[] upperBound, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, targetValues);

	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterface.ObjectiveFunction, double[], double[], double[], double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters,	double[] lowerBound, double[] upperBound, double[] parameterStep, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, targetValues);

	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction, double[], double[], double[], double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters, double[] lowerBound, double[] upperBound, double[] parameterSteps, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, targetValues);
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction, double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters, double[] targetValues) {

		OptimizerInterface optimizer = null;

		switch (optimizerType) {
		case BroydenFletcherGoldfarbShanno:
			if(targetValues.length > 1) throw new IllegalArgumentException();
			optimizer = new BroydenFletcherGoldfarbShanno(initialParameters, targetValues[0], maxIterations, errorTolerance) {

				@Override
				public void setValues(double[] parameters, double[] values) throws SolverException {
					objectiveFunction.setValues(parameters, values);
				}

				@Override
				public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
					objectiveFunction.setDerivatives(parameters, derivatives);
				}
			};
			break;
		case GradientDescentArmijo:
			if(targetValues.length > 1) throw new IllegalArgumentException();
			optimizer = new GradientDescentArmijosRule(initialParameters, targetValues[0], errorTolerance, maxIterations) {

				@Override
				public void setValues(double[] parameters, double[] values) throws SolverException {
					objectiveFunction.setValues(parameters, values);
				}

				@Override
				public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
					objectiveFunction.setDerivatives(parameters, derivatives);
				}
			};
			break;
		case SimpleGradientDescent:
			if(targetValues.length > 1) throw new IllegalArgumentException();
			optimizer = new SimpleGradientDescent(initialParameters, targetValues[0], errorTolerance, maxIterations) {

				@Override
				public void setValues(double[] parameters, double[] values) throws SolverException {
					objectiveFunction.setValues(parameters, values);
				}

				@Override
				public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
					objectiveFunction.setDerivatives(parameters, derivatives);
				}
			};
			break;
		case TruncatedGaussNetwonForUnderdeterminedNSLP:
			if(targetValues.length > 1) throw new IllegalArgumentException();
			optimizer = new TruncatedGaussNewtonForUnterdetNSLP(initialParameters, targetValues[0], errorTolerance, maxIterations) {

				@Override
				public void setValues(double[] parameters, double[] values) throws SolverException {
					objectiveFunction.setValues(parameters, values);
				}

				@Override
				public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
					objectiveFunction.setDerivatives(parameters, derivatives);
				}
			};
			break;
		case LevenbergMarquardt:
			optimizer = (new LevenbergMarquardt(initialParameters, targetValues, maxIterations, maxThreads) {

				@Override
				public void setValues(double[] parameters, double[] values) throws SolverException {
					objectiveFunction.setValues(parameters, values);
				}

				@Override
				public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
					objectiveFunction.setDerivatives(parameters, derivatives);
				}
			}).setErrorTolerance(errorTolerance);
			break;
		default:
			throw new UnknownError();		
		}

		return optimizer;
	}

	/* (non-Javadoc)
	 * @see net.finmath.optimizer.OptimizerFactoryInterface#getOptimizer(net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction, double[], double[], double[], double[])
	 */
	@Override
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters, double[] lowerBound, double[] upperBound, double[] targetValues) {
		return getOptimizer(objectiveFunction, initialParameters, targetValues);
	}
}
