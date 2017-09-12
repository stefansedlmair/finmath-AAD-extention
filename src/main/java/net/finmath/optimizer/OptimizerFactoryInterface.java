/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christianfries.com.
 *
 * Created on 29.05.2015
 */

package net.finmath.optimizer;

import net.finmath.optimizer.OptimizerInterface.ObjectiveFunction;
import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;

/**
 * @author Christian Fries
 *
 */
public interface OptimizerFactoryInterface {

	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters, double[] targetValues);
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters, double[] lowerBound,double[]  upperBound, double[] targetValues);
	public OptimizerInterface getOptimizer(ObjectiveFunction objectiveFunction, double[] initialParameters, double[] lowerBound,double[]  upperBound, double[] parameterStep, double[] targetValues);
	
	
	/** enables to change the derivations technique from the standard Optimizer interface
	 * 
	 * @see DerivativeFunction
	 * */
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters, double[] lowerBound, double[] upperBound, double[] parameterSteps, double[] targetValues);
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters, double[] targetValues);
	public OptimizerInterface getOptimizer(DerivativeFunction objectiveFunction, double[] initialParameters, double[] lowerBound, double[] upperBound, double[] targetValues);
}
