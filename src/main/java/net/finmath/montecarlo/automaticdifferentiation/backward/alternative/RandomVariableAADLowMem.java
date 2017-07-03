/**
 * 
 */
package net.finmath.montecarlo.automaticdifferentiation.backward.alternative;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.stream.DoubleStream;

import net.finmath.functions.DoubleTernaryOperator;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.RandomVariable;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * Implementation of <code>RandomVariableInterface</code> having the additional feature to calculate the backward algorithmic differentiation.
 * 
 * For construction use the factory method <code>constructNewAADRandomVariable</code>.
 *
 * @author Stefan Sedlmair
 * @version 2.5
 */
public class RandomVariableAADLowMem implements RandomVariableDifferentiableInterface {

	private static final long serialVersionUID = 2459373647785530657L;
	
	private static AtomicLong randomVariableUID = new AtomicLong(0);

	private static AbstractRandomVariableFactory constantsFactory = new RandomVariableFactory();
	
	/* static elements of the class are shared between all members */
	public static enum OperatorType {
		ADD, MULT, DIV, SUB, SQUARED, SQRT, LOG, SIN, COS, EXP, INVERT, CAP, FLOOR, ABS, 
		ADDPRODUCT, ADDRATIO, SUBRATIO, BARRIER, DISCOUNT, ACCURUE, POW, AVERAGE, VARIANCE, 
		STDEV, MIN, MAX, STDERROR, SVARIANCE, AVERAGE2, STDEV2, VARIANCE2, STDERROR2
	}

	/* index of corresponding random variable in the static array list*/
	private final RandomVariableInterface ownRandomVariable;
	private final long ownRandomVariableUID;

	/* this could maybe be outsourced to own class ParentElement */
	private final ArrayList<RandomVariableInterface> arguments;
	private final OperatorType parentOperator;
	private ArrayList<Long> childUIDs;
	private boolean isConstant;

	private RandomVariableAADLowMem(RandomVariableInterface ownRandomVariable, ArrayList<RandomVariableInterface> arguments, OperatorType parentOperator, 
			ArrayList<Long> childUIDs ,boolean isConstant) {
		super();
		this.ownRandomVariable 		= ownRandomVariable;
		this.arguments 				= arguments;
		this.parentOperator 		= parentOperator;
		this.childUIDs 				= childUIDs;
		this.isConstant 			= isConstant;
		
		this.ownRandomVariableUID 	= randomVariableUID.getAndIncrement();
	}

	public RandomVariableAADLowMem(RandomVariableInterface ownRandomVariable) {
		this(ownRandomVariable, null, null, new ArrayList<Long>(), false);
	}
	
	public RandomVariableAADLowMem(double time, double[] values) {
		this(constantsFactory.createRandomVariable(time, values), null, null, new ArrayList<Long>(), false);
	}
	
	public RandomVariableAADLowMem(double time, double value) {
		this(constantsFactory.createRandomVariable(time, value), null, null, new ArrayList<Long>(), false);
	}
	
	public RandomVariableAADLowMem(double value) {
		this(constantsFactory.createRandomVariable(value), null, null, new ArrayList<Long>(), false);
	}
	
	private RandomVariableAADLowMem(OperatorType parentOperator, RandomVariableInterface[] arguments) {
		this.ownRandomVariableUID 	= randomVariableUID.getAndIncrement();		
		
		ArrayList<RandomVariableInterface> argumentArrayList = new ArrayList<>();
		for(RandomVariableInterface arg : arguments){
			argumentArrayList.add(arg);
			if(arg instanceof RandomVariableAADLowMem) ((RandomVariableAADLowMem) arg).addChild(ownRandomVariableUID);
		}
		
		this.ownRandomVariable 		= null;
		this.arguments 				= argumentArrayList;
		this.parentOperator 		= parentOperator;
		this.childUIDs 				= new ArrayList<>();
		this.isConstant 			= false;

	}

	private RandomVariableInterface apply(OperatorType operator, RandomVariableInterface[] arguments){


		RandomVariableInterface resultrandomvariable = null;
		RandomVariableInterface X = arguments.length > 0 ? valuesOf(arguments[0]) : null;
		RandomVariableInterface Y = arguments.length > 1 ? valuesOf(arguments[1]) : null;
		RandomVariableInterface Z = arguments.length > 2 ? valuesOf(arguments[2]) : null;

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
				resultrandomvariable = constantsFactory.createRandomVariable(X.getAverage());
				break;
			case STDERROR:
				resultrandomvariable = constantsFactory.createRandomVariable(X.getStandardError());
				break;
			case STDEV:
				resultrandomvariable = constantsFactory.createRandomVariable(X.getStandardDeviation());
				break;
			case VARIANCE:
				resultrandomvariable = constantsFactory.createRandomVariable(X.getVariance());
				break;
			case SVARIANCE:
				resultrandomvariable = constantsFactory.createRandomVariable(X.getSampleVariance());
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
				resultrandomvariable = constantsFactory.createRandomVariable(X.getAverage(Y));
				break;
			case STDERROR2:
				resultrandomvariable = constantsFactory.createRandomVariable(X.getStandardError(Y));
				break;
			case STDEV2:
				resultrandomvariable = constantsFactory.createRandomVariable(X.getStandardDeviation(Y));
				break;
			case VARIANCE2:
				resultrandomvariable = constantsFactory.createRandomVariable(X.getVariance(Y));
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
			
			
//		ArrayList<RandomVariableInterface> argumentsArrayList = new ArrayList<>();
//		for(RandomVariableInterface arg:arguments) argumentsArrayList.add(arg);
//		
//		/* create new RandomVariable */			
//		RandomVariableAADLowMem newRandomVariableAAD = new RandomVariableAADLowMem(operator, argumentsArrayList);
//	
//		/* add new variable as child to its parents */
//		for(RandomVariableInterface argument : argumentsArrayList)
//			if(argument instanceof RandomVariableAADLowMem)
//				((RandomVariableAADLowMem) argument).addChild(newRandomVariableAAD.getID());
//		
//		/* return new RandomVariable */
//		return newRandomVariableAAD;
			
			return resultrandomvariable;
	}

	public String toString(){
		return  super.toString() + "\n" + 
				"time:              " + getFiltrationTime() + "\n" + 
				"realizations:      " + Arrays.toString(getRealizations()) + "\n" + 
				"randomVariableUID: " + getID() + "\n" +
				"parentIDs:         " + Arrays.toString(getParentRandomVariableUIDs()) + ((getArguments() == null) ? "" : (" type: " + parentOperator.name())) + "\n" +
				"isTrueVariable:    " + isVariable() + "\n";
	}

	private RandomVariableInterface partialDerivativeWithRespectTo(long variableIndex){
		
		int posInArguments = getArgumentUIDs().indexOf(variableIndex);
		
		/* if random variable not dependent on variable or it is constant anyway return 0.0 */
		if(posInArguments < 0 || isConstant) return constantsFactory.createRandomVariable(0.0);

		RandomVariableInterface resultrandomvariable = null;
		RandomVariableInterface X = getArguments().size() > 0 ? valuesOf(getArguments().get(0)) : null;
		RandomVariableInterface Y = getArguments().size() > 1 ? valuesOf(getArguments().get(1)) : null;
		RandomVariableInterface Z = getArguments().size() > 2 ? valuesOf(getArguments().get(2)) : null;
		
		boolean isFirstArgument 	= posInArguments == 0;
		boolean isSecondArgument 	= posInArguments == 1;
		
			switch(parentOperator){
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
				resultrandomvariable = constantsFactory.createRandomVariable(X.size()).invert();
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
				resultrandomvariable = constantsFactory.createRandomVariable(1.0);
				break;
			case SUB:
				resultrandomvariable = constantsFactory.createRandomVariable(isFirstArgument ? 1.0 : -1.0);
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
				resultrandomvariable = isFirstArgument ? Y.mult(X.pow(Y.getAverage() - 1.0)) : constantsFactory.createRandomVariable(0.0);

			case ADDPRODUCT:
				if(isFirstArgument){
					resultrandomvariable = constantsFactory.createRandomVariable(1.0);
				} else if(isSecondArgument){
					resultrandomvariable = Z;
				} else {
					resultrandomvariable = Y;
				}
				break;
			case ADDRATIO:
				if(isFirstArgument){
					resultrandomvariable = constantsFactory.createRandomVariable(1.0);
				} else if(isSecondArgument){
					resultrandomvariable = Z.invert();
				} else {
					resultrandomvariable = Y.div(Z.squared());
				}
				break;
			case SUBRATIO:
				if(isFirstArgument){
					resultrandomvariable = constantsFactory.createRandomVariable(1.0);
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
					resultrandomvariable = X.barrier(X, constantsFactory.createRandomVariable(1.0), constantsFactory.createRandomVariable(0.0));
				} else {
					resultrandomvariable = X.barrier(X, constantsFactory.createRandomVariable(0.0), constantsFactory.createRandomVariable(1.0));
				}
			default:
				break;
		}
		

		return resultrandomvariable;
	}

	/**
	 * Implements the AAD Algorithm
	 * @return HashMap where the key is the internal index of the random variable with respect to which the partial derivative was computed. This key then gives access to the actual derivative.
	 * */
	public Map<Long, RandomVariableInterface> getGradient(){

		/* get dependence tree */
		TreeMap<Long, RandomVariableAADLowMem> mapOfDependentRandomVariables = mapAllDependentRandomVariableAADv2s();
		
		/* key set is indicating in which order random variables were generated */
		Set<Long> idsOfDependentRandomVariables = mapOfDependentRandomVariables.keySet();
		
		int numberOfDependentRandomVariables = idsOfDependentRandomVariables.size();

		/* catch trivial case here */
		if(numberOfDependentRandomVariables == 1) 
			return new HashMap<Long, RandomVariableInterface>()
				{{put(getID(), constantsFactory.createRandomVariable(getFiltrationTime(), isConstant() ? 0.0 : 1.0));}};
		
			
		/*_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_*/
		
		Map<Long, RandomVariableInterface> gradient = new HashMap<Long, RandomVariableInterface>();
		Map<Long, RandomVariableInterface> omegaHat = new HashMap<Long, RandomVariableInterface>();

		/* first entry (with highest variable UID) of omegaHat is set to 1.0 */
		long lastVariableIndex = mapOfDependentRandomVariables.lastKey();
		
		omegaHat.put(lastVariableIndex, constantsFactory.createRandomVariable(1.0));
		
		if(mapOfDependentRandomVariables.get(lastVariableIndex).isVariable())
			gradient.put(lastVariableIndex, omegaHat.get(lastVariableIndex));
		
		for(long variableIndex : mapOfDependentRandomVariables.descendingKeySet().tailSet(lastVariableIndex, false)){
			
			RandomVariableInterface newOmegaHatEntry = constantsFactory.createRandomVariable(0.0);
			
			for(long functionIndex : mapOfDependentRandomVariables.get(variableIndex).getChildrenUIDs()){
				RandomVariableInterface D_i_j = mapOfDependentRandomVariables.get(functionIndex).partialDerivativeWithRespectTo(variableIndex);
				newOmegaHatEntry = newOmegaHatEntry.addProduct(D_i_j, omegaHat.get(functionIndex));
			}
			
			if(mapOfDependentRandomVariables.get(variableIndex).isVariable())
				gradient.put(variableIndex, newOmegaHatEntry);
			
			omegaHat.put(variableIndex, newOmegaHatEntry);
		}

		return gradient;
	}

	/* for all functions that need to be differentiated and are returned as double in the Interface, write a method to return it as RandomVariableAAD 
	 * that is deterministic by its nature. For their double-returning pendant just return the average of the deterministic RandomVariableAAD  */

	public RandomVariableInterface getAverageAsRandomVariableAAD(RandomVariableInterface probabilities){
		/*returns deterministic AAD random variable */
		return new RandomVariableAADLowMem(OperatorType.AVERAGE2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface getVarianceAsRandomVariableAAD(RandomVariableInterface probabilities){
		/*returns deterministic AAD random variable */
		return new RandomVariableAADLowMem(OperatorType.VARIANCE2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface 	getStandardDeviationAsRandomVariableAAD(RandomVariableInterface probabilities){
		/*returns deterministic AAD random variable */
		return new RandomVariableAADLowMem(OperatorType.STDEV2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface 	getStandardErrorAsRandomVariableAAD(RandomVariableInterface probabilities){
		/*returns deterministic AAD random variable */
		return new RandomVariableAADLowMem(OperatorType.STDERROR2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface getAverageAsRandomVariableAAD(){
		/*returns deterministic AAD random variable */
		return new RandomVariableAADLowMem(OperatorType.AVERAGE, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface getVarianceAsRandomVariableAAD(){
		/*returns deterministic AAD random variable */
		return new RandomVariableAADLowMem(OperatorType.VARIANCE, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface getSampleVarianceAsRandomVariableAAD() {
		/*returns deterministic AAD random variable */
		return new RandomVariableAADLowMem(OperatorType.SVARIANCE, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface 	getStandardDeviationAsRandomVariableAAD(){
		/*returns deterministic AAD random variable */
		return new RandomVariableAADLowMem(OperatorType.STDEV, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface getStandardErrorAsRandomVariableAAD(){
		/*returns deterministic AAD random variable */
		return new RandomVariableAADLowMem(OperatorType.STDERROR, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface 	getMinAsRandomVariableAAD(){
		/*returns deterministic AAD random variable */
		return new RandomVariableAADLowMem(OperatorType.MIN, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface 	getMaxAsRandomVariableAAD(){
		/*returns deterministic AAD random variable */
		return new RandomVariableAADLowMem(OperatorType.MAX, new RandomVariableInterface[]{this});
	}

	/* setter and getter */

	private boolean isConstant(){
		return isConstant;
	}

	private boolean isVariable() {
		return (isConstant() == false && getArguments() == null);
	}	

	public void setIsConstantTo(boolean isConstant){
		this.isConstant = isConstant;
	}

	private RandomVariableInterface valuesOf(RandomVariableInterface rv){
		if(rv instanceof RandomVariableAADLowMem){
			RandomVariableAADLowMem rvAADlm = (RandomVariableAADLowMem) rv;
			return rvAADlm.ownRandomVariable != null ? rvAADlm.ownRandomVariable : apply(rvAADlm.parentOperator, rvAADlm.getArguments().toArray(new RandomVariableInterface[getArguments().size()]));
		} else {
			return rv;
		}
	}

	private ArrayList<RandomVariableInterface> getArguments(){
		return arguments;
	}
	
	private ArrayList<RandomVariableAADLowMem> getDifferentiableArgumentsOnly(){
		ArrayList<RandomVariableAADLowMem> differentiableArguments = new ArrayList<>();
		if(getArguments() != null){
			for(RandomVariableInterface arg : getArguments())
				if(arg instanceof RandomVariableAADLowMem)
					differentiableArguments.add((RandomVariableAADLowMem)arg);
		}
		return differentiableArguments;
	}
	
	private void addChild(long childUID){
		getChildrenUIDs().add(childUID);
	}
	
	@Override
	public Long getID() {
		return ownRandomVariableUID;
	}
	
	
	private long[] getParentRandomVariableUIDs(){
		long[] parentUIDs = new long[getDifferentiableArgumentsOnly().size()];
		
		for(int i = 0; i < parentUIDs.length; i++)
			parentUIDs[i] = getDifferentiableArgumentsOnly().get(i).getID();
		
		return parentUIDs;
	}
	
	private ArrayList<Long> getChildrenUIDs(){
		return childUIDs;
	}
	
	private ArrayList<Long> getArgumentUIDs(){
		ArrayList<Long> arguemntUIDs = new ArrayList<>();
		for(RandomVariableInterface argument : getArguments()) 
			arguemntUIDs.add((long) (argument instanceof RandomVariableAADLowMem ? ((RandomVariableDifferentiableInterface) argument).getID() : Long.MAX_VALUE));
		return arguemntUIDs;
	}
	
	/* get the dependence tree for a instance of RandomVariableAADv2 */
	private TreeMap<Long, RandomVariableAADLowMem> mapAllDependentRandomVariableAADv2s(){
		TreeMap<Long, RandomVariableAADLowMem> mapOfDependenRandomVariableAAD = new TreeMap<Long, RandomVariableAADLowMem>();
		
		/* add the variable it self */
		if(!mapOfDependenRandomVariableAAD.containsKey(getID())) mapOfDependenRandomVariableAAD.put(getID(), this);
		
		for(RandomVariableAADLowMem argument:getDifferentiableArgumentsOnly())
			mapOfDependenRandomVariableAAD.putAll(argument.mapAllDependentRandomVariableAADv2s());
		
		return mapOfDependenRandomVariableAAD;
	}
		
	/*--------------------------------------------------------------------------------------------------------------------------------------------------*/



	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#equals(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public boolean equals(RandomVariableInterface randomVariable) {
		return valuesOf(this).equals(randomVariable);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getFiltrationTime()
	 */
	@Override
	public double getFiltrationTime() {
		return valuesOf(this).getFiltrationTime();
	}



	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#get(int)
	 */
	@Override
	public double get(int pathOrState) {
		return valuesOf(this).get(pathOrState);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#size()
	 */
	@Override
	public int size() {
		return valuesOf(this).size();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#isDeterministic()
	 */
	@Override
	public boolean isDeterministic() {
		return valuesOf(this).isDeterministic();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getRealizations()
	 */
	@Override
	public double[] getRealizations() {
		return valuesOf(this).getRealizations();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getRealizations(int)
	 */
	@Override
	public double[] getRealizations(int numberOfPaths) {
		return valuesOf(this).getRealizations(numberOfPaths);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getMin()
	 */
	@Override
	public double getMin() {
		return valuesOf(getMinAsRandomVariableAAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getMax()
	 */
	@Override
	public double getMax() {
		return valuesOf(getMaxAsRandomVariableAAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getAverage()
	 */
	@Override
	public double getAverage() {		
		return valuesOf(getAverageAsRandomVariableAAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getAverage(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getAverage(RandomVariableInterface probabilities) {
		return valuesOf(getAverageAsRandomVariableAAD(probabilities)).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getVariance()
	 */
	@Override
	public double getVariance() {
		return valuesOf(getVarianceAsRandomVariableAAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getVariance(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getVariance(RandomVariableInterface probabilities) {
		return valuesOf(getAverageAsRandomVariableAAD(probabilities)).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getSampleVariance()
	 */
	@Override
	public double getSampleVariance() {
		return valuesOf(getSampleVarianceAsRandomVariableAAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardDeviation()
	 */
	@Override
	public double getStandardDeviation() {
		return valuesOf(getStandardDeviationAsRandomVariableAAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardDeviation(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getStandardDeviation(RandomVariableInterface probabilities) {
		return valuesOf(getStandardDeviationAsRandomVariableAAD(probabilities)).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardError()
	 */
	@Override
	public double getStandardError() {
		return valuesOf(getStandardErrorAsRandomVariableAAD()).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardError(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getStandardError(RandomVariableInterface probabilities) {
		return valuesOf(getStandardErrorAsRandomVariableAAD(probabilities)).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getQuantile(double)
	 */
	@Override
	public double getQuantile(double quantile) {
		return valuesOf(this).getQuantile(quantile);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getQuantile(double, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getQuantile(double quantile, RandomVariableInterface probabilities) {
		return valuesOf(this).getQuantile(quantile, probabilities);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getQuantileExpectation(double, double)
	 */
	@Override
	public double getQuantileExpectation(double quantileStart, double quantileEnd) {
		return valuesOf(this).getQuantileExpectation(quantileStart, quantileEnd);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getHistogram(double[])
	 */
	@Override
	public double[] getHistogram(double[] intervalPoints) {
		return valuesOf(this).getHistogram(intervalPoints);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getHistogram(int, double)
	 */
	@Override
	public double[][] getHistogram(int numberOfPoints, double standardDeviations) {
		return valuesOf(this).getHistogram(numberOfPoints, standardDeviations);
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
		return new RandomVariableAADLowMem(OperatorType.CAP, new RandomVariableInterface[]{this, new RandomVariableAADLowMem(cap)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#floor(double)
	 */
	@Override
	public RandomVariableInterface floor(double floor) {
		return new RandomVariableAADLowMem(OperatorType.FLOOR, new RandomVariableInterface[]{this, new RandomVariableAADLowMem(floor)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#add(double)
	 */
	@Override
	public RandomVariableInterface add(double value) {
		return new RandomVariableAADLowMem(OperatorType.ADD, new RandomVariableInterface[]{this, new RandomVariableAADLowMem(value)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sub(double)
	 */
	@Override
	public RandomVariableInterface sub(double value) {
		return new RandomVariableAADLowMem(OperatorType.SUB, new RandomVariableInterface[]{this, new RandomVariableAADLowMem(value)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#mult(double)
	 */
	@Override
	public RandomVariableInterface mult(double value) {
		return new RandomVariableAADLowMem(OperatorType.MULT, new RandomVariableInterface[]{this, new RandomVariableAADLowMem(value)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#div(double)
	 */
	@Override
	public RandomVariableInterface div(double value) {
		return new RandomVariableAADLowMem(OperatorType.DIV, new RandomVariableInterface[]{this, new RandomVariableAADLowMem(value)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#pow(double)
	 */
	@Override
	public RandomVariableInterface pow(double exponent) {
		return new RandomVariableAADLowMem(OperatorType.POW, new RandomVariableInterface[]{this, new RandomVariableAADLowMem(exponent)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#squared()
	 */
	@Override
	public RandomVariableInterface squared() {
		return new RandomVariableAADLowMem(OperatorType.SQUARED, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sqrt()
	 */
	@Override
	public RandomVariableInterface sqrt() {
		return new RandomVariableAADLowMem(OperatorType.SQRT, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#exp()
	 */
	@Override
	public RandomVariableInterface exp() {
		return new RandomVariableAADLowMem(OperatorType.EXP, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#log()
	 */
	@Override
	public RandomVariableInterface log() {
		return new RandomVariableAADLowMem(OperatorType.LOG, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sin()
	 */
	@Override
	public RandomVariableInterface sin() {
		return new RandomVariableAADLowMem(OperatorType.SIN, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#cos()
	 */
	@Override
	public RandomVariableInterface cos() {
		return new RandomVariableAADLowMem(OperatorType.COS, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#add(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface add(RandomVariableInterface randomVariable) {	
		return new RandomVariableAADLowMem(OperatorType.ADD, new RandomVariableInterface[]{this, randomVariable});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sub(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface sub(RandomVariableInterface randomVariable) {
		return new RandomVariableAADLowMem(OperatorType.SUB, new RandomVariableInterface[]{this, randomVariable});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#mult(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface mult(RandomVariableInterface randomVariable) {
		return new RandomVariableAADLowMem(OperatorType.MULT, new RandomVariableInterface[]{this, randomVariable});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#div(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface div(RandomVariableInterface randomVariable) {
		return new RandomVariableAADLowMem(OperatorType.DIV, new RandomVariableInterface[]{this, randomVariable});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#cap(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface cap(RandomVariableInterface cap) {
		return new RandomVariableAADLowMem(OperatorType.CAP, new RandomVariableInterface[]{this, cap});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#floor(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface floor(RandomVariableInterface floor) {
		return new RandomVariableAADLowMem(OperatorType.FLOOR, new RandomVariableInterface[]{this, floor});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#accrue(net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface accrue(RandomVariableInterface rate, double periodLength) {
		return new RandomVariableAADLowMem(OperatorType.ACCURUE, new RandomVariableInterface[]{this, rate, new RandomVariableAADLowMem(periodLength)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#discount(net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface discount(RandomVariableInterface rate, double periodLength) {
		return new RandomVariableAADLowMem(OperatorType.DISCOUNT, new RandomVariableInterface[]{this, rate, new RandomVariableAADLowMem(periodLength)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#barrier(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface barrier(RandomVariableInterface trigger,
			RandomVariableInterface valueIfTriggerNonNegative, RandomVariableInterface valueIfTriggerNegative) {
		return new RandomVariableAADLowMem(OperatorType.BARRIER, new RandomVariableInterface[]{this, valueIfTriggerNonNegative, valueIfTriggerNegative});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#barrier(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface barrier(RandomVariableInterface trigger,
			RandomVariableInterface valueIfTriggerNonNegative, double valueIfTriggerNegative) {
		return new RandomVariableAADLowMem(OperatorType.BARRIER, new RandomVariableInterface[]{this, valueIfTriggerNonNegative, new RandomVariableAADLowMem(valueIfTriggerNegative)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#invert()
	 */
	@Override
	public RandomVariableInterface invert() {
		return new RandomVariableAADLowMem(OperatorType.INVERT, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#abs()
	 */
	@Override
	public RandomVariableInterface abs() {
		return new RandomVariableAADLowMem(OperatorType.ABS, new RandomVariableInterface[]{this});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addProduct(net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface addProduct(RandomVariableInterface factor1, double factor2) {
		return new RandomVariableAADLowMem(OperatorType.ADDPRODUCT, new RandomVariableInterface[]{this, factor1, new RandomVariableAADLowMem(factor2)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addProduct(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface addProduct(RandomVariableInterface factor1, RandomVariableInterface factor2) {
		return new RandomVariableAADLowMem(OperatorType.ADDPRODUCT, new RandomVariableInterface[]{this, factor1, factor2});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addRatio(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface addRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
		return new RandomVariableAADLowMem(OperatorType.ADDRATIO, new RandomVariableInterface[]{this, numerator, denominator});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#subRatio(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface subRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
		return new RandomVariableAADLowMem(OperatorType.SUBRATIO, new RandomVariableInterface[]{this, numerator, denominator});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#isNaN()
	 */
	@Override
	public RandomVariableInterface isNaN() {
		return valuesOf(this).isNaN();
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
		return valuesOf(this).getOperator();
	}

	@Override
	public DoubleStream getRealizationsStream() {
		return valuesOf(this).getRealizationsStream();
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
