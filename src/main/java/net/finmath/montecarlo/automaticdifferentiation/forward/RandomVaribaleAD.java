/**
 * 
 */
package net.finmath.montecarlo.automaticdifferentiation.forward;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.stream.DoubleStream;

import net.finmath.functions.DoubleTernaryOperator;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.optimizer.SolverException;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * Implementation of the {@link RandomVariableDifferentiableInterface} using the the forward mode of 
 * the automatic differentiation algorithm.
 * 
 * @author Stefan Sedlmair
 *
 * @version 0.1
 */
public class RandomVaribaleAD implements RandomVariableDifferentiableInterface {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6428797769639959963L;
	private static enum OperatorType {
		ADD, MULT, DIV, SUB, SQUARED, SQRT, LOG, SIN, COS, EXP, INVERT, CAP, FLOOR, ABS, 
		ADDPRODUCT, ADDRATIO, SUBRATIO, BARRIER, DISCOUNT, ACCURUE, POW, MIN, MAX, AVERAGE, VARIANCE, 
		STDEV, STDERROR, SVARIANCE, AVERAGE2, VARIANCE2, 
		STDEV2, STDERROR2
	}

	private static AtomicLong nextRandomVariableUID = new AtomicLong(0);

	private final RandomVariableInterface randomVariable;
	private final long randomVariableUID;

	private final List<RandomVariableInterface> arguments;
	private OperatorType operator;
	private boolean isConstant;

	/*
	 * factory for the production of random variables that are not for
	 * associated with the AD algorithm, thus offline.
	 * 
	 * TODO: can the factory be declared non static?
	 */
	private static AbstractRandomVariableFactory offlineFactory = new RandomVariableFactory();
	private static boolean useMultiThreading = true;
	
	/** private general construction assigns every new instance a unique id */
	private RandomVaribaleAD(RandomVariableInterface randomVariableInterface, List<RandomVariableInterface> arguments, OperatorType operator, boolean isConstant) {
		this.randomVariable = randomVariableInterface;
		this.arguments = arguments;
		this.operator = operator;
		this.isConstant = isConstant;

		this.randomVariableUID = nextRandomVariableUID.getAndIncrement();
	}

	public RandomVaribaleAD(double time, double[] values) {
		this(offlineFactory.createRandomVariable(time, values), null, null, false);
	}

	public RandomVaribaleAD(double time, double value) {
		this(offlineFactory.createRandomVariable(time, value), null, null, false);
	}
	
	public RandomVaribaleAD(double value) {
		this(offlineFactory.createRandomVariable(value), null, null, false);
	}
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see net.finmath.montecarlo.automaticdifferentiation.
	 * RandomVariableDifferentiableInterface#getID()
	 */
	@Override
	public Long getID() {
		return randomVariableUID;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see net.finmath.montecarlo.automaticdifferentiation.
	 * RandomVariableDifferentiableInterface#getGradient()
	 */
	@Override
	public Map<Long, RandomVariableInterface> getGradient() {

		TreeMap<Long, RandomVariableInterface> gradient = new TreeMap<Long, RandomVariableInterface>();
		
		if(!useMultiThreading){
			for(Long key : getKeySetOfDependentVariables())
				gradient.put(key, getGradientFor(key));
		} else {
			
			int numberOfThreads = Runtime.getRuntime().availableProcessors();
			ExecutorService executor = Executors.newFixedThreadPool(numberOfThreads);
			
			TreeSet<Long> keySetOfDependentVariables = getKeySetOfDependentVariables();
			TreeMap<Long, Future<RandomVariableInterface>> futureGradient = new TreeMap<Long, Future<RandomVariableInterface>>();
			
			// submit all tasks to executor
			for(Long key : keySetOfDependentVariables){
				
				Callable<RandomVariableInterface> worker = new Callable<RandomVariableInterface>() {
					public RandomVariableInterface call() {
						 return getGradientFor(key);
					}
				};
		
				futureGradient.put(key, executor.submit(worker));
			}
			
			for(Long key : futureGradient.keySet())
				try {
					gradient.put(key, futureGradient.get(key).get());
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (ExecutionException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
		}
		
		return gradient;
	}
	
	public RandomVariableInterface getGradientFor(long variableIndex){
		
		if(getID() == variableIndex) return offlineFactory.createRandomVariable(getFiltrationTime(), 1.0);
		
		RandomVariableInterface gradient = offlineFactory.createRandomVariable(getFiltrationTime(), 0.0);
		
		if(getArguments() != null){
			for(RandomVariableInterface arg : getArguments()){
				if(arg instanceof RandomVaribaleAD){
					RandomVaribaleAD argAD = (RandomVaribaleAD) arg;
					gradient = gradient.addProduct(partialDerivativeWithRespectTo(argAD), argAD.getGradientFor(variableIndex));
				}
			}	
		}
		
		return gradient;
	}
	
	private TreeSet<Long> getKeySetOfDependentVariables() {
		TreeSet<Long> keySetForGradient = new TreeSet<Long>();
		
		if(getArguments() != null){
			for(RandomVariableInterface arg : getArguments()){
				if(arg instanceof RandomVaribaleAD){
					RandomVaribaleAD argAD = (RandomVaribaleAD) arg;
					if(argAD.isTrueVariable())
						keySetForGradient.add(argAD.getID());
					else 
						keySetForGradient.addAll(argAD.getKeySetOfDependentVariables());
				}
			}
		}
		
		return keySetForGradient;
	}
	
	private boolean isTrueVariable(){
		return (getArguments() == null && isConstant() == false);
	}

	private boolean isConstant() {
		return isConstant;
	}

	private RandomVariableInterface valuesOf(RandomVariableInterface randomVariable){
		if(randomVariable instanceof RandomVaribaleAD) 	return ((RandomVaribaleAD) randomVariable).getValues();
		else 												return randomVariable;
	}
	
	private RandomVariableInterface apply(OperatorType operator, RandomVariableInterface[] arguments){
		
		RandomVariableInterface X = arguments.length > 0 ? valuesOf(arguments[0]) : null;
		RandomVariableInterface Y = arguments.length > 1 ? valuesOf(arguments[1]) : null;
		RandomVariableInterface Z = arguments.length > 2 ? valuesOf(arguments[2]) : null;

		RandomVariableInterface resultrandomvariable = null;
		
		switch(operator){
		case SQUARED:
			resultrandomvariable = X.squared();
			break;
		case SQRT:
			resultrandomvariable = X.sqrt();
			break;
		case EXP:
			resultrandomvariable = X.exp();
			break;
		case LOG:
			resultrandomvariable = X.log();
			break;
		case SIN:
			resultrandomvariable = X.sin();
			break;
		case COS:
			resultrandomvariable = X.cos();
			break;
		case ABS:
			resultrandomvariable = X.abs();
			break;
		case INVERT:
			resultrandomvariable = X.invert();
			break;
		case AVERAGE:
			resultrandomvariable = offlineFactory.createRandomVariable(X.getAverage());
			break;
		case STDERROR:
			resultrandomvariable = offlineFactory.createRandomVariable(X.getStandardError());
			break;
		case STDEV:
			resultrandomvariable = offlineFactory.createRandomVariable(X.getStandardDeviation());
			break;
		case VARIANCE:
			resultrandomvariable = offlineFactory.createRandomVariable(X.getVariance());
			break;
		case SVARIANCE:
			resultrandomvariable = offlineFactory.createRandomVariable(X.getSampleVariance());
			break;
		case ADD:
			resultrandomvariable = X.add(Y);
			break;
		case SUB:
			resultrandomvariable = X.sub(Y);
			break;
		case MULT:
			resultrandomvariable = X.mult(Y);
			break;
		case DIV:
			resultrandomvariable = X.div(Y);
			break;
		case CAP:
			resultrandomvariable = Y.isDeterministic() ? X.cap(Y.getAverage()) : X.cap(Y);
			break;
		case FLOOR:
			resultrandomvariable = Y.isDeterministic() ? X.floor(Y.getAverage()) : X.floor(Y);
			break;			
		case POW:
			resultrandomvariable = X.pow( /* argument is deterministic random variable */ Y.getAverage());
			break;
		case AVERAGE2:
			resultrandomvariable = offlineFactory.createRandomVariable(X.getAverage(Y));
			break;
		case STDERROR2:
			resultrandomvariable = offlineFactory.createRandomVariable(X.getStandardError(Y));
			break;
		case STDEV2:
			resultrandomvariable = offlineFactory.createRandomVariable(X.getStandardDeviation(Y));
			break;
		case VARIANCE2:
			resultrandomvariable = offlineFactory.createRandomVariable(X.getVariance(Y));
			break;
		case ADDPRODUCT:
			resultrandomvariable = X.addProduct(Y,Z);
			break;
		case ADDRATIO:
			resultrandomvariable = X.addRatio(Y, Z);
			break;
		case SUBRATIO:
			resultrandomvariable = X.subRatio(Y, Z);
			break;
		case ACCURUE:
			resultrandomvariable = X.accrue(Y, /* second argument is deterministic anyway */ Z.getAverage());
			break;
		case DISCOUNT:
			resultrandomvariable = X.discount(Y, /* second argument is deterministic anyway */ Z.getAverage());
			break;
		default:
			throw new IllegalArgumentException();
		}
	/* create new RandomVariableAADv2 which is definitely NOT Constant */
			RandomVaribaleAD newRandomVariableAD = new RandomVaribaleAD(resultrandomvariable, toList(arguments), operator,/*not constant*/ false);

	/* return new RandomVariable */
	return newRandomVariableAD;		
	}
	
	private RandomVariableInterface partialDerivativeWithRespectTo(RandomVariableDifferentiableInterface variable){

		int placeInArguments = getArguments().indexOf(variable);
		
		if(placeInArguments < 0) offlineFactory.createRandomVariable(variable.getFiltrationTime(), 0.0);
		
		RandomVariableInterface X = getArguments().size() > 0 ? valuesOf(getArguments().get(0)) : null;
		RandomVariableInterface Y = getArguments().size() > 1 ? valuesOf(getArguments().get(1)) : null;
		RandomVariableInterface Z = getArguments().size() > 2 ? valuesOf(getArguments().get(2)) : null;

		RandomVariableInterface resultrandomvariable = null;
		
		boolean isFirstArgument = placeInArguments == 0;
		boolean isSecondArgument = placeInArguments == 1;
		
			switch(operator){
			/* functions with one argument  */
			case SQUARED:
				resultrandomvariable = X.mult(2.0);
				break;
			case SQRT:
				resultrandomvariable = X.sqrt().invert().mult(0.5);
				break;
			case EXP:
				resultrandomvariable = X.exp();
				break;
			case LOG:
				resultrandomvariable = X.invert();
				break;
			case SIN:
				resultrandomvariable = X.cos();
				break;
			case COS:
				resultrandomvariable = X.sin().mult(-1.0);
				break;
			case AVERAGE:
				resultrandomvariable = offlineFactory.createRandomVariable(X.size()).invert();
				break;
			case VARIANCE:
				resultrandomvariable = X.sub(X.getAverage()*(2.0*X.size()-1.0)/X.size()).mult(2.0/X.size());
				break;
			case STDEV:
				resultrandomvariable = X.sub(X.getAverage()*(2.0*X.size()-1.0)/X.size()).mult(2.0/X.size()).mult(0.5).div(Math.sqrt(X.getVariance()));
				break;
			case MIN:
				resultrandomvariable = X.apply(x -> (x == X.getMin()) ? 1.0 : 0.0);
				break;
			case MAX:
				resultrandomvariable = X.apply(x -> (x == X.getMax()) ? 1.0 : 0.0);
				break;
			case ABS:
				resultrandomvariable = X.apply(x -> (x > 0.0) ? 1.0 : (x < 0) ? -1.0 : 0.0);
				break;
			case STDERROR:
				resultrandomvariable = X.sub(X.getAverage()*(2.0*X.size()-1.0)/X.size()).mult(2.0/X.size()).mult(0.5).div(Math.sqrt(X.getVariance() * X.size()));
				break;
			case SVARIANCE:
				resultrandomvariable = X.sub(X.getAverage()*(2.0*X.size()-1.0)/X.size()).mult(2.0/(X.size()-1));
				break;
			case ADD:
				resultrandomvariable = offlineFactory.createRandomVariable(1.0);
				break;
			case SUB:
				resultrandomvariable = offlineFactory.createRandomVariable(isFirstArgument ? 1.0 : -1.0);
				break;
			case MULT:
				resultrandomvariable = isFirstArgument ? Y : X;
				break;
			case DIV:
				resultrandomvariable = isFirstArgument ? Y.invert() : X.div(Y.squared());
				break;
			case CAP:
				if(isFirstArgument)
					resultrandomvariable = Y.isDeterministic() ? X.apply(x -> (x > Y.getAverage()) ? 0.0 : 1.0) : X.apply((x,y) -> (x > y) ? 0.0 : 1.0, Y);
				else
					resultrandomvariable = X.isDeterministic() ? Y.apply(y -> (y < X.getAverage()) ? 1.0 : 0.0) : Y.apply((y,x) -> (y < x) ? 1.0 : 0.0, X);
				break;
			case FLOOR:
				if(isFirstArgument)
					resultrandomvariable = Y.isDeterministic() ? X.apply(x -> (x > Y.getAverage()) ? 1.0 : 0.0) : X.apply((x,y) -> (x > y) ? 1.0 : 0.0, Y);
				else
					resultrandomvariable = X.isDeterministic() ? Y.apply(y -> (y < X.getAverage()) ? 0.0 : 1.0) : Y.apply((y,x) -> (y < x) ? 0.0 : 1.0, X);
				break;
			case AVERAGE2:
				resultrandomvariable = isFirstArgument ? Y : X;
				break;
			case VARIANCE2:
				resultrandomvariable = isFirstArgument ? Y.mult(2.0).mult(X.mult(Y.add(X.getAverage(Y)*(X.size()-1)).sub(X.getAverage(Y)))) :
					X.mult(2.0).mult(Y.mult(X.add(Y.getAverage(X)*(X.size()-1)).sub(Y.getAverage(X))));
				break;
			case STDEV2:				
				resultrandomvariable = isFirstArgument ? Y.mult(2.0).mult(X.mult(Y.add(X.getAverage(Y)*(X.size()-1)).sub(X.getAverage(Y)))).div(Math.sqrt(X.getVariance(Y))) :
				X.mult(2.0).mult(Y.mult(X.add(Y.getAverage(X)*(X.size()-1)).sub(Y.getAverage(X)))).div(Math.sqrt(Y.getVariance(X)));
				break;
			case STDERROR2:				
				resultrandomvariable = isFirstArgument ? Y.mult(2.0).mult(X.mult(Y.add(X.getAverage(Y)*(X.size()-1)).sub(X.getAverage(Y)))).div(Math.sqrt(X.getVariance(Y) * X.size())) :
				X.mult(2.0).mult(Y.mult(X.add(Y.getAverage(X)*(X.size()-1)).sub(Y.getAverage(X)))).div(Math.sqrt(Y.getVariance(X) * Y.size()));
				break;
			case POW:
				/* second argument will always be deterministic and constant! */
				resultrandomvariable = isFirstArgument ? Y.mult(X.pow(Y.getAverage() - 1.0)) : offlineFactory.createRandomVariable(0.0);
			
			case ADDPRODUCT:
				if(isFirstArgument){
					resultrandomvariable = offlineFactory.createRandomVariable(1.0);
				} else if(isSecondArgument){
					resultrandomvariable = Z;
				} else {
					resultrandomvariable = Y;
				}
				break;
			case ADDRATIO:
				if(isFirstArgument){
					resultrandomvariable = offlineFactory.createRandomVariable(1.0);
				} else if(isSecondArgument){
					resultrandomvariable = Z.invert();
				} else {
					resultrandomvariable = Y.div(Z.squared());
				}
				break;
			case SUBRATIO:
				if(isFirstArgument){
					resultrandomvariable = offlineFactory.createRandomVariable(1.0);
				} else if(isSecondArgument){
					resultrandomvariable = Z.invert().mult(-1.0);
				} else {
					resultrandomvariable = Y.div(Z.squared()).mult(-1.0);
				}
				break;
			case ACCURUE:
				if(isFirstArgument){
					resultrandomvariable = Y.mult(Z).add(1.0);
				} else if(isSecondArgument){
					resultrandomvariable = X.mult(Z);
				} else {
					resultrandomvariable = X.mult(Y);
				}
				break;
			case DISCOUNT:
				if(isFirstArgument){
					resultrandomvariable = Y.mult(Z).add(1.0).invert();
				} else if(isSecondArgument){
					resultrandomvariable = X.mult(Z).div(Y.mult(Z).add(1.0).squared());
				} else {
					resultrandomvariable = X.mult(Y).div(Y.mult(Z).add(1.0).squared());
				}
				break;
			case BARRIER:
				if(isFirstArgument){
					resultrandomvariable = X.apply(x -> (x == 0.0) ? Double.POSITIVE_INFINITY : 0.0);
				} else if(isSecondArgument){
					resultrandomvariable = X.barrier(X, offlineFactory.createRandomVariable(1.0), offlineFactory.createRandomVariable(0.0));
				} else {
					resultrandomvariable = X.barrier(X, offlineFactory.createRandomVariable(0.0), offlineFactory.createRandomVariable(1.0));
				}
			default:
				break;
			}

		return resultrandomvariable;
	}
	

	private List<RandomVariableInterface> getArguments(){
		return arguments;
	}
	
	private List<RandomVariableInterface> toList(RandomVariableInterface[] rvArray){
		List<RandomVariableInterface> newList = new ArrayList<>();
		for(RandomVariableInterface rv:rvArray) newList.add(rv);
		return newList;
	}

	private RandomVariableInterface getValues(){
		return randomVariable;
	}
	
	public static void useMultiThreading(boolean useMultiThreading){
		RandomVaribaleAD.useMultiThreading = useMultiThreading;
	}
	
	/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	/* for all functions that need to be differentiated and are returned as double in the Interface, write a method to return it as RandomVariableAAD 
	 * that is deterministic by its nature. For their double-returning pendant just return the average of the deterministic RandomVariableAAD  */

	public RandomVariableInterface getAverageAsRandomVariableAD(RandomVariableInterface probabilities){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.AVERAGE2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface getVarianceAsRandomVariableAD(RandomVariableInterface probabilities){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.VARIANCE2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface 	getStandardDeviationAsRandomVariableAD(RandomVariableInterface probabilities){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.STDEV2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface 	getStandardErrorAsRandomVariableAD(RandomVariableInterface probabilities){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.STDERROR2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface getAverageAsRandomVariableAD(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.AVERAGE, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface getVarianceAsRandomVariableAD(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.VARIANCE, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface getSampleVarianceAsRandomVariableAD() {
		/*returns deterministic AAD random variable */
		return apply(OperatorType.SVARIANCE, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface 	getStandardDeviationAsRandomVariableAD(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.STDEV, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface getStandardErrorAsRandomVariableAD(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.STDERROR, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface 	getMinAsRandomVariableAD(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.MIN, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface 	getMaxAsRandomVariableAD(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.MAX, new RandomVariableInterface[]{this});
	}

	/* setter and getter */


	public void setIsConstantTo(boolean isConstant){
		this.isConstant = isConstant;
	}


	
	/*--------------------------------------------------------------------------------------------------------------------------------------------------*/



	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#equals(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public boolean equals(RandomVariableInterface randomVariable) {
		return getValues().equals(randomVariable);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getFiltrationTime()
	 */
	@Override
	public double getFiltrationTime() {
		return getValues().getFiltrationTime();
	}



	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#get(int)
	 */
	@Override
	public double get(int pathOrState) {
		return getValues().get(pathOrState);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#size()
	 */
	@Override
	public int size() {
		return getValues().size();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#isDeterministic()
	 */
	@Override
	public boolean isDeterministic() {
		return getValues().isDeterministic();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getRealizations()
	 */
	@Override
	public double[] getRealizations() {
		return getValues().getRealizations();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getRealizations(int)
	 */
	@Override
	public double[] getRealizations(int numberOfPaths) {
		return getValues().getRealizations(numberOfPaths);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getMin()
	 */
	@Override
	public double getMin() {
		return valuesOf(getMinAsRandomVariableAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getMax()
	 */
	@Override
	public double getMax() {
		return  valuesOf(getMaxAsRandomVariableAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getAverage()
	 */
	@Override
	public double getAverage() {		
		return  valuesOf(getAverageAsRandomVariableAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getAverage(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getAverage(RandomVariableInterface probabilities) {
		return valuesOf(getAverageAsRandomVariableAD(probabilities)).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getVariance()
	 */
	@Override
	public double getVariance() {
		return valuesOf(getVarianceAsRandomVariableAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getVariance(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getVariance(RandomVariableInterface probabilities) {
		return valuesOf(getVarianceAsRandomVariableAD(probabilities)).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getSampleVariance()
	 */
	@Override
	public double getSampleVariance() {
		return valuesOf(getSampleVarianceAsRandomVariableAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardDeviation()
	 */
	@Override
	public double getStandardDeviation() {
		return valuesOf(getStandardDeviationAsRandomVariableAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardDeviation(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getStandardDeviation(RandomVariableInterface probabilities) {
		return valuesOf(getStandardDeviationAsRandomVariableAD(probabilities)).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardError()
	 */
	@Override
	public double getStandardError() {
		return valuesOf(getStandardErrorAsRandomVariableAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardError(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getStandardError(RandomVariableInterface probabilities) {
		return valuesOf(getStandardErrorAsRandomVariableAD(probabilities)).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getQuantile(double)
	 */
	@Override
	public double getQuantile(double quantile) {
		return getValues().getQuantile(quantile);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getQuantile(double, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getQuantile(double quantile, RandomVariableInterface probabilities) {
		return getValues().getQuantile(quantile, probabilities);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getQuantileExpectation(double, double)
	 */
	@Override
	public double getQuantileExpectation(double quantileStart, double quantileEnd) {
		return getValues().getQuantileExpectation(quantileStart, quantileEnd);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getHistogram(double[])
	 */
	@Override
	public double[] getHistogram(double[] intervalPoints) {
		return getValues().getHistogram(intervalPoints);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getHistogram(int, double)
	 */
	@Override
	public double[][] getHistogram(int numberOfPoints, double standardDeviations) {
		return getValues().getHistogram(numberOfPoints, standardDeviations);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#cache()
	 */
	@Override
	public RandomVariableInterface cache() {
		return this;
	}

	@Override
	public RandomVariableInterface cap(double cap) {
		return apply(OperatorType.CAP, new RandomVariableInterface[]{this, new RandomVaribaleAD(cap)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#floor(double)
	 */
	@Override
	public RandomVariableInterface floor(double floor) {
		return apply(OperatorType.FLOOR, new RandomVariableInterface[]{this, new RandomVaribaleAD(floor)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#add(double)
	 */
	@Override
	public RandomVariableInterface add(double value) {
		return apply(OperatorType.ADD, new RandomVariableInterface[]{this, new RandomVaribaleAD(value)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sub(double)
	 */
	@Override
	public RandomVariableInterface sub(double value) {
		return apply(OperatorType.SUB, new RandomVariableInterface[]{this, new RandomVaribaleAD(value)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#mult(double)
	 */
	@Override
	public RandomVariableInterface mult(double value) {
		return apply(OperatorType.MULT, new RandomVariableInterface[]{this, new RandomVaribaleAD(value)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#div(double)
	 */
	@Override
	public RandomVariableInterface div(double value) {
		return apply(OperatorType.DIV, new RandomVariableInterface[]{this, new RandomVaribaleAD(value)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#pow(double)
	 */
	@Override
	public RandomVariableInterface pow(double exponent) {
		return apply(OperatorType.POW, new RandomVariableInterface[]{this, new RandomVaribaleAD(exponent)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#squared()
	 */
	@Override
	public RandomVariableInterface squared() {
		return apply(OperatorType.SQUARED, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sqrt()
	 */
	@Override
	public RandomVariableInterface sqrt() {
		return apply(OperatorType.SQRT, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#exp()
	 */
	@Override
	public RandomVariableInterface exp() {
		return apply(OperatorType.EXP, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#log()
	 */
	@Override
	public RandomVariableInterface log() {
		return apply(OperatorType.LOG, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sin()
	 */
	@Override
	public RandomVariableInterface sin() {
		return apply(OperatorType.SIN, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#cos()
	 */
	@Override
	public RandomVariableInterface cos() {
		return apply(OperatorType.COS, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#add(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface add(RandomVariableInterface randomVariable) {	
		return apply(OperatorType.ADD, new RandomVariableInterface[]{this, randomVariable});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sub(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface sub(RandomVariableInterface randomVariable) {
		return apply(OperatorType.SUB, new RandomVariableInterface[]{this, randomVariable});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#mult(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface mult(RandomVariableInterface randomVariable) {
		return apply(OperatorType.MULT, new RandomVariableInterface[]{this, randomVariable});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#div(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface div(RandomVariableInterface randomVariable) {
		return apply(OperatorType.DIV, new RandomVariableInterface[]{this, randomVariable});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#cap(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface cap(RandomVariableInterface cap) {
		return apply(OperatorType.CAP, new RandomVariableInterface[]{this, cap});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#floor(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface floor(RandomVariableInterface floor) {
		return apply(OperatorType.FLOOR, new RandomVariableInterface[]{this, floor});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#accrue(net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface accrue(RandomVariableInterface rate, double periodLength) {
		return apply(OperatorType.ACCURUE, new RandomVariableInterface[]{this, rate, new RandomVaribaleAD(periodLength)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#discount(net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface discount(RandomVariableInterface rate, double periodLength) {
		return apply(OperatorType.DISCOUNT, new RandomVariableInterface[]{this, rate, new RandomVaribaleAD(periodLength)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#barrier(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface barrier(RandomVariableInterface trigger,
			RandomVariableInterface valueIfTriggerNonNegative, RandomVariableInterface valueIfTriggerNegative) {
		return apply(OperatorType.BARRIER, new RandomVariableInterface[]{this, valueIfTriggerNonNegative, valueIfTriggerNegative});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#barrier(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface barrier(RandomVariableInterface trigger,
			RandomVariableInterface valueIfTriggerNonNegative, double valueIfTriggerNegative) {
		return apply(OperatorType.BARRIER, new RandomVariableInterface[]{this, valueIfTriggerNonNegative, new RandomVaribaleAD(valueIfTriggerNegative)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#invert()
	 */
	@Override
	public RandomVariableInterface invert() {
		return apply(OperatorType.INVERT, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#abs()
	 */
	@Override
	public RandomVariableInterface abs() {
		return apply(OperatorType.ABS, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addProduct(net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface addProduct(RandomVariableInterface factor1, double factor2) {
		return apply(OperatorType.ADDPRODUCT, new RandomVariableInterface[]{this, factor1, new RandomVaribaleAD(factor2)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addProduct(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface addProduct(RandomVariableInterface factor1, RandomVariableInterface factor2) {
		return apply(OperatorType.ADDPRODUCT, new RandomVariableInterface[]{this, factor1, factor2});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addRatio(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface addRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
		return apply(OperatorType.ADDRATIO, new RandomVariableInterface[]{this, numerator, denominator});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#subRatio(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface subRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
		return apply(OperatorType.SUBRATIO, new RandomVariableInterface[]{this, numerator, denominator});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#isNaN()
	 */
	@Override
	public RandomVariableInterface isNaN() {
		return getValues().isNaN();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getMutableCopy()
	 */
	@Override
	public RandomVariableInterface getMutableCopy() {
		return this;
	}

	@Override
	public IntToDoubleFunction getOperator() {
		throw new UnsupportedOperationException("Not supported.");
	}

	@Override
	public DoubleStream getRealizationsStream() {
		throw new UnsupportedOperationException("Not supported.");
	}

	@Override
	public RandomVariableInterface apply(DoubleUnaryOperator operator) {
		throw new UnsupportedOperationException("Applying functions is not supported.");
	}

	@Override
	public RandomVariableInterface apply(DoubleBinaryOperator operator, RandomVariableInterface argument) {
		throw new UnsupportedOperationException("Applying functions is not supported.");
	}

	@Override
	public RandomVariableInterface apply(DoubleTernaryOperator operator, RandomVariableInterface argument1, RandomVariableInterface argument2) {
		throw new UnsupportedOperationException("Applying functions is not supported.");
	}
}
