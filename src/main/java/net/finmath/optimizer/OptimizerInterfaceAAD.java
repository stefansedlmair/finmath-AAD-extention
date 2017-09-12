/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christianfries.com.
 *
 * Created on 29.05.2015
 */

package net.finmath.optimizer;

/**
 * @author Stefan Sedlmair
 *
 */
public interface OptimizerInterfaceAAD extends OptimizerInterface {

	public interface DerivativeFunction extends ObjectiveFunction {
	
		public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException;
	
	}
}