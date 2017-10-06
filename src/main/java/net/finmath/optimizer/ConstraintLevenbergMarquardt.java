/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 16.06.2006
 */
package net.finmath.optimizer;

import java.util.Arrays;

/**
 *  Extension of the LevenbergMarquardt Class taking parameter boundaries into account. 
 *  
 *  The class is implemented such that the internal parameters are unbounded ,i.e. element of the real numbers.
 *  <ul>
 *  <li>When calculating the values the internal parameters are transformed via a transformation function
 *  	<ul>
 *  	<li> for $|a|<\infty$ and $|b|<\infty$: 
 *  		\[f(x) = \frac{b-a}{1+e^{-x}} + a\] </li>
 *  	<li> for $|a|=\infty$ and $|b|<\infty$: 
 *  		\[f(x) = \log(1+\exp(x)) + b\] </li>
 *  	<li> for $|a|<\infty$ and $|b|=\infty$: 
 *  		\[f(x) = -\log(1+\exp(x)) + a\] </li>
 *  	<li> for $|a|=\infty$ and $|b|=\infty$: 
 *  		\[f(x) = x\] </li>
 *  	</ul>
 *  </li>
 *  <li>When calculating the gradient the class uses the chain rule to first calculate the derivatives with respect to the bounded parameters and then multiplies the derivative of the transformation function to the result.
 *  	<ul>
 *  	<li> for $|a|<\infty$ and $|b|<\infty$: 
 *  		\[f'(x) = \frac{(b-a)e^{-x}}{(1+e^{-x})^2}\] </li>
 *  	<li> for $|a|=\infty$ and $|b|<\infty$: 
 *  		\[f(x) = \frac{e^x}{1+e^x} \] </li>
 *  	<li> for $|a|<\infty$ and $|b|=\infty$: 
 *  		\[f(x) = -\frac{e^x}{1+e^x} \] </li>
 *  	<li> for $|a|=\infty$ and $|b|=\infty$: 
 *  		\[f(x) = 1\] </li>
 *  	</ul>
 *  </li>
 *  </ul>
 * */
public abstract class ConstraintLevenbergMarquardt extends LevenbergMarquardt{

	private static final long serialVersionUID = 4406296875085574008L;

	public ConstraintLevenbergMarquardt(double[] initialParameters, double[] targetValues, int maxIterations, int maxThreads) {
		super(initialParameters, targetValues, maxIterations, maxThreads);
		
		this.boundedInitialParameters = initialParameters;
		this.numberOfParameters = initialParameters.length;
	}
	
	private 	  int numberOfParameters;
	private final double[] boundedInitialParameters;
	
	private double[] lowerBound = null;
	private double[] upperBound = null;

	public ConstraintLevenbergMarquardt setBounds(double[] lowerBound, double[] upperBound) {
		if(done()) throw new UnsupportedOperationException("Solver cannot be modified after it has run.");
		if(numberOfParameters != lowerBound.length) throw new IllegalArgumentException("lowerBound has to be equal the length of parameter");
		if(numberOfParameters != upperBound.length) throw new IllegalArgumentException("upperBound has to be equal the length of parameter");
		this.upperBound = upperBound;
		this.lowerBound = lowerBound;

		double[] internalInitialParameters = transformationFromBoundaries(boundedInitialParameters, lowerBound, upperBound);

		return (ConstraintLevenbergMarquardt) this.setInitialParameters(internalInitialParameters);
	}
	
	public double[] getLowerBound() {
		if(lowerBound == null) {
			lowerBound = new double[numberOfParameters];
			Arrays.fill(lowerBound, Double.NEGATIVE_INFINITY);
		}
		return lowerBound;
	}
	
	public double[] getUpperBound() {
		if(upperBound == null) {
			upperBound = new double[numberOfParameters];
			Arrays.fill(upperBound, Double.POSITIVE_INFINITY);
		}
		return upperBound;
	}
	
	@Override
	public double[] getBestFitParameters() {
		double[] bestFitInternalParameters = super.getBestFitParameters();
		if(bestFitInternalParameters == null) 
			return null;
		else 
			return transformationIntoBoundaries(super.getBestFitParameters(), getLowerBound(), getUpperBound());
	}
	
	public abstract void setValuesFunction(double[] parameters, double[] values) throws SolverException;
	public 			void setDerivativesFunction(double[] parameters, double[][] derivatives) throws SolverException {
		super.setDerivatives(parameters, derivatives);
	}
	
	@Override
	public void setValues(double[] parameters, double[] values) throws SolverException {
			final double[] parameterInBounds = transformationIntoBoundaries(parameters, getLowerBound(), getUpperBound());
			setValuesFunction(parameterInBounds, values);
	}
	
	@Override
	public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
		
//		final double[] parameterInBounds = transformationIntoBoundaries(parameters, getLowerBound(), getUpperBound());
		setDerivativesFunction(parameters, derivatives);
		
//		double[] derivativesFromBounderies = transformDerivative(parameterInBounds, getLowerBound(), getUpperBound());
//		
//		for(int parameterIndex = 0; parameterIndex < derivatives.length; parameterIndex++) {
//			for(int functionIndex = 0; functionIndex < derivatives[parameterIndex].length; functionIndex++) {
//				derivatives[parameterIndex][functionIndex] *= derivativesFromBounderies[parameterIndex];
//			}
//		}
	}
	
	public double[] transformationIntoBoundaries(double[] parameters, double[] lowerBounds, double[] upperBounds) {
		synchronized (this) {
			double[] parametersInBound = new double[numberOfParameters];
			
			for(int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
				double lowerBound = lowerBounds[parameterIndex];
				double upperBound = upperBounds[parameterIndex];
				double parameter = parameters[parameterIndex];
				double parameterInBound = 0.0;
				
				boolean lowerBoundIsFinite = Double.isFinite(lowerBound);
				boolean upperBoundIsFinite = Double.isFinite(upperBound);
				
				if(lowerBoundIsFinite && upperBoundIsFinite) {
					// f: \mathbb{R}\rightarrow (a,b) ; f(x) = \frac{b-a}{1+e^{-x}} + a 
					parameterInBound = (upperBound - lowerBound)/(1+Math.exp(-parameter)) + lowerBound;
				} else
				if(lowerBoundIsFinite && !upperBoundIsFinite) {
					// f: \mathbb{R}\rightarrow (a,\infty) ; f(x) = \log(1+\exp(x)) + a
					parameterInBound = Math.log1p(Math.exp(parameter)) + lowerBound;
//					parameterInBound = Math.max(parameter, 0.0);
				} else
				if(!lowerBoundIsFinite && upperBoundIsFinite) {
					// f: \mathbb{R}\rightarrow (\infty,b) ; f(x) = -\log(1+\exp(x)) + b
					parameterInBound = -Math.log(1.0 + Math.exp(parameter)) + upperBound;
				} else 
				if(!lowerBoundIsFinite && !upperBoundIsFinite){
					// f: \mathbb{R}\rightarrow (\infty,\infty) ; f(x) = x 
					parameterInBound = parameter;
				}
				
				parametersInBound[parameterIndex] = parameterInBound;
			}
			
			return parametersInBound;
		}
	}
	
	public double[] transformDerivative(double[] parameters, double[] lowerBounds, double[] upperBounds) {
		int numberOfParameters = parameters.length;
		
		double[] parametersfromBound = new double[numberOfParameters];
		
		for(int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
			double lowerBound = lowerBounds[parameterIndex];
			double upperBound = upperBounds[parameterIndex];
			double parameter = parameters[parameterIndex];
			double parameterfromBound = 0.0;
			
			boolean lowerBoundIsFinite = Double.isFinite(lowerBound);
			boolean upperBoundIsFinite = Double.isFinite(upperBound);
			
			if(lowerBoundIsFinite && upperBoundIsFinite) {
				//f'(x) = \frac{(b-a)e^{-x}}{(1+e^{-x})^2}
				double p = Math.exp(-parameter);
				parameterfromBound = (upperBound - lowerBound)*p/Math.pow(1.0+p, 2.0);
			} else
			if(lowerBoundIsFinite && !upperBoundIsFinite) {
				// f'(x) = \frac{e^x}{1+e^x} 
				double p = Math.exp(parameter);
				parameterfromBound = p/(1.0+p);
			} else
			if(!lowerBoundIsFinite && upperBoundIsFinite) {
				// f'(x) = -\frac{e^x}{1+e^x} 
				double p = Math.exp(parameter);
				parameterfromBound = -p/(1.0+p);
			} else 
			if(!lowerBoundIsFinite && !upperBoundIsFinite){
				//  f'(x) = x 
				parameterfromBound = 1.0;
			}
			
			parametersfromBound[parameterIndex] = parameterfromBound;
		}
		
		return parametersfromBound;
	}
	
	public double[] transformationFromBoundaries(double[] parameters, double[] lowerBounds, double[] upperBounds) {
		synchronized (this) {
			double[] parametersInBound = new double[numberOfParameters];
			
			for(int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
				double lowerBound = lowerBounds[parameterIndex];
				double upperBound = upperBounds[parameterIndex];
				double parameter = parameters[parameterIndex];
				double parameterInBound = 0.0;
				
				boolean lowerBoundIsFinite = Double.isFinite(lowerBound);
				boolean upperBoundIsFinite = Double.isFinite(upperBound);
				
				if(lowerBoundIsFinite && upperBoundIsFinite) {
					// f: \mathbb{R}\rightarrow (a,b) ; f(x) = \frac{b-a}{1+e^{-x}} + a 
					parameterInBound = Math.log((parameter-lowerBound)/(upperBound-parameter));
				} else
				if(lowerBoundIsFinite && !upperBoundIsFinite) {
					// f: \mathbb{R}\rightarrow (a,\infty) ; f(x) = \log(1+\exp(x)) + a
					parameterInBound = Math.log(Math.exp(parameter - lowerBound) - 1.0);
//					parameterInBound = Math.max(parameter, 0.0);
				} else
				if(!lowerBoundIsFinite && upperBoundIsFinite) {
					// f: \mathbb{R}\rightarrow (\infty,b) ; f(x) = -\log(1+\exp(x)) + b
					parameterInBound =  Math.log(Math.exp(upperBound - parameter) - 1.0);
				} else 
				if(!lowerBoundIsFinite && !upperBoundIsFinite){
					// f: \mathbb{R}\rightarrow (\infty,\infty) ; f(x) = x 
					parameterInBound = parameter;
				}
				
				parametersInBound[parameterIndex] = parameterInBound;
			}
			
			return parametersInBound;
		}
	}
}
