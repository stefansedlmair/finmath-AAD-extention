 /**
 * 
 */
package net.finmath.rootfinder;

import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;

import net.finmath.functions.LinearAlgebra;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * @author Stefan Sedlmair
 * @version 0.1
 */
public class LevenbergMarquardtSolver implements RandomVariableDifferentiableMultiDimensionalRootFinderInterface {

	private final TreeMap<Long, RandomVariableInterface> initialValue; /* \beta_0 		*/
	
	private TreeMap<Long, RandomVariableInterface>	nextParameterSet;	/* \beta_i 		*/	// Stores the next point to be returned by getPoint()
	private TreeMap<Long, RandomVariableInterface>	bestParameterSet;	/* \beta_{best} */
		
	private int			numberOfIterations		= 0;           								// Number of numberOfIterations
	private int 		maxNumberOfIterations 	= Integer.MAX_VALUE;
	private boolean		isDone					= false;             						// Will be true if machine accuracy has been reached
	private double 		accuracy 				= Double.MAX_VALUE;		
	private double		targetAccuracy			= 0.0;
	
	private double			 	lambda	 			= 0.001;
	private final double		lambdaDivisor		= 1.3;
	private final double		lambdaMultiplicator	= 2.0;
	
	/* predefine how to generate random variables */
	private AbstractRandomVariableFactory nonDifferentiableRandomVariableFactory = new RandomVariableFactory();
	
	/* predefine variables to  */
	private final RandomVariableInterface targetFunctionValue;						/* y 			*/
	private final RandomVariableInterface uncertainties;							/* \sigma 		*/
	
	/*multithreaded option - TODO: slower than serial implementation! WHY? */
	private boolean useMultithreading = false;

	public LevenbergMarquardtSolver(TreeMap<Long, RandomVariableInterface> initialValue, RandomVariableInterface targetFunctionValue,
			RandomVariableInterface uncertainties) {
		super();
		this.initialValue = initialValue;
		
		this.nextParameterSet = initialValue;
		this.bestParameterSet = initialValue;
		
		this.targetFunctionValue = targetFunctionValue;
		this.uncertainties = uncertainties;
	}
	
	public LevenbergMarquardtSolver(TreeMap<Long, RandomVariableInterface> initialValue, RandomVariableInterface targetFunctionValue){
		super();
		this.initialValue = initialValue;
		
		this.nextParameterSet = initialValue;
		this.bestParameterSet = initialValue;
		
		this.targetFunctionValue = targetFunctionValue;
		this.uncertainties = nonDifferentiableRandomVariableFactory.createRandomVariable(1.0);
	}
	
	public LevenbergMarquardtSolver(TreeMap<Long, RandomVariableInterface> initialValue, RandomVariableInterface targetFunctionValue,
			double targetAccuracy, int maxNumberOfIterations) {
		this(initialValue, targetFunctionValue);
		this.targetAccuracy 		= targetAccuracy;
		this.maxNumberOfIterations 	= maxNumberOfIterations;
	}

	public LevenbergMarquardtSolver(TreeMap<Long, RandomVariableInterface> initialValue, RandomVariableInterface targetFunctionValue,
			RandomVariableInterface uncertainties, double targetAccuracy, int maxNumberOfIterations) {
		this(initialValue, targetFunctionValue, uncertainties);
		this.targetAccuracy 		= targetAccuracy;
		this.maxNumberOfIterations 	= maxNumberOfIterations;
	}
	
	/**
	 * implements 
	 * \[ 
	 * \left[ J^TJ +\lambda\cdot\text{diag}(J^TJ)\right]\delta = J^T\left[y - f(x,\beta) \right]
	 * \]
	 * and solves the equation for $\delta$.
	 * 
	 * @param gradient map of all partial derivative with respect to all dependent random variables, whose IDs are the keys in this map
	 * @param lambda parameter associated with the step size of the iteration
	 * */
	private Map<Long, RandomVariableInterface> estimateDelta(RandomVariableInterface currentFunctionValue, Map<Long, RandomVariableInterface> gradient){
			
		// set up variables
		int numberOfVariables = gradient.size();
		int numberOfRealizations = currentFunctionValue.size();
		
		// change from Maps to Arrays for solving the linear equation system
		RandomVariableInterface[] gradientArray = new RandomVariableInterface[numberOfVariables];
					
		for(int variableIndex = 0; variableIndex < numberOfVariables; variableIndex++){
			long key = gradient.keySet().iterator().next();
			gradientArray[variableIndex] 	= gradient.get(key);
		}
		
		RandomVariableInterface currentError = targetFunctionValue.sub(currentFunctionValue).div(uncertainties);
		
		double[][][] A = new double[numberOfRealizations][numberOfVariables][numberOfVariables];
		double[][] b = new double[numberOfRealizations][numberOfVariables];
		
		/* A is symmetric thus only calculate the upper half */
		for(int xIndex = 0; xIndex < numberOfVariables; xIndex++){			
			for(int yIndex = xIndex; yIndex < numberOfVariables; yIndex++){
		
				
				RandomVariableInterface alpha = gradientArray[xIndex].mult(gradientArray[yIndex]).mult(xIndex == yIndex ? 1.0 + lambda : 1.0);	
				
				for(int zIndex = 0; zIndex < numberOfRealizations; zIndex++){
					A[zIndex][xIndex][yIndex] = alpha.get(zIndex);
					A[zIndex][yIndex][xIndex] = alpha.get(zIndex);
				}
			}

			/* b = J^T\left[y - f(x,\beta) \right]*/
			RandomVariableInterface beta = gradientArray[xIndex].mult(currentError);
			for(int zIndex = 0; zIndex < numberOfRealizations; zIndex++){
				b[zIndex][xIndex] = beta.get(zIndex);
			}
		}
		
		double[][] deltaArray = new double[numberOfRealizations][numberOfVariables];
		
		
		// serial implementation
		for(int zIndex = 0; zIndex < numberOfRealizations; zIndex++)
			deltaArray[zIndex] = LinearAlgebra.solveLinearEquation(A[zIndex], b[zIndex]);

		
		// re-associate the deltas with their parameter ids
		TreeMap<Long, RandomVariableInterface> delta = new TreeMap<>();
		int variableIndex = 0;
		for(Long key:gradient.keySet()){
			
			double[] deltaPerRealizations = new double[numberOfRealizations];
			for(int zIndex = 0; zIndex < numberOfRealizations; zIndex++)
				deltaPerRealizations[zIndex] = deltaArray[zIndex][variableIndex];			
			
			delta.put(key, nonDifferentiableRandomVariableFactory.createRandomVariable(gradient.get(key).getFiltrationTime(), 
					/* take average over all suggested parameter updates from each realization */
					nonDifferentiableRandomVariableFactory.createRandomVariable(0.0, deltaPerRealizations).getAverage()));
			
			variableIndex++;
		}
		
		return delta;
	}
	/* (non-Javadoc)
	 * @see net.finmath.rootfinder.RandomVariableRootFinderUsingDerivative#setValueAndDerivative()
	 */
	@Override
    public void setValueAndDerivative(RandomVariableInterface currentFunctionValue, Map<Long, RandomVariableInterface> gradient) {
    	
    	double currentAccuracy = targetFunctionValue.sub(currentFunctionValue).div(uncertainties).squared()
    			.getAverage(nonDifferentiableRandomVariableFactory.createRandomVariable(1.0));
    	
    	if(currentAccuracy < getAccuracy())
		{
			accuracy			= currentAccuracy;
			bestParameterSet	= nextParameterSet;
			
			// decrease step size to approach minimum
			lambda /= lambdaDivisor;
		} else {
			// increase step size to look for better place on surface
			lambda *= lambdaMultiplicator;
		}
    	
    	// estimate the delta for each parameter
    	Map<Long, RandomVariableInterface> delta = estimateDelta(currentFunctionValue, gradient);
    	
    	// add the delta to each associated parameter
    	TreeMap<Long, RandomVariableInterface> newNextParameterSet = new TreeMap<>();
    	Iterator<Long> nextParameterKeySetIterator = nextParameterSet.keySet().iterator();
    	Iterator<Long> deltaKeySetIterator = delta.keySet().iterator();
    	
    	while(deltaKeySetIterator.hasNext() && nextParameterKeySetIterator.hasNext()){
    		Long deltaKey = deltaKeySetIterator.next();
    		Long nextParameterKey = nextParameterKeySetIterator.next();
    		
    		newNextParameterSet.put(deltaKey, nextParameterSet.get(nextParameterKey).add(delta.get(deltaKey)));
    	}
//    	
//    	for(Long key : nextParameterSet.keySet())
//    		newNextParameterSet.put(((RandomVariableDifferentiableInterface) delta.get(key)).getID(), nextParameterSet.get(key).add(delta.get(key)));
//    	
    	nextParameterSet = newNextParameterSet;
    	
    	numberOfIterations++;
    	
    	isDone = getAccuracy() < targetAccuracy || getNumberOfIterations() >= maxNumberOfIterations || Double.isInfinite(getLambda());
	}

	/* (non-Javadoc)
	 * @see net.finmath.rootfinder.RandomVariableRootFinderUsingDerivative#getNumberOfIterations()
	 */
	@Override
	public int getNumberOfIterations() {
		return numberOfIterations;
	}

	/* (non-Javadoc)
	 * @see net.finmath.rootfinder.RandomVariableRootFinderUsingDerivative#getAccuracy()
	 */
	@Override
	public double getAccuracy() {
		return accuracy;
	}
	
	public double getLambda(){
		return lambda;
	}

	/* (non-Javadoc)
	 * @see net.finmath.rootfinder.RandomVariableRootFinderUsingDerivative#isDone()
	 */
	@Override
	public boolean isDone() {
		return isDone;
	}

	@Override
	public TreeMap<Long, RandomVariableInterface> getNextParameters() {
		return nextParameterSet;
	}


	@Override
	public TreeMap<Long, RandomVariableInterface> getBestParameters() {
		return bestParameterSet;
	}


}
