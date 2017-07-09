/**
 * 
 */
package net.finmath.montecarlo.automaticdifferentiation.backward;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import net.finmath.functions.DoubleTernaryOperator;
import net.finmath.montecarlo.RandomVariable;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * Implementation of <code>RandomVariableDifferentiableInterface</code> using
 * the backward algorithmic differentiation (adjoint algorithmic differentiation, AAD).
 * 
 * @author Christian Fries
 * @author Stefan Sedlmair
 * @version 1.0
 */
public class RandomVariableDifferentiableAAD2 implements RandomVariableDifferentiableInterface {

	private static final long serialVersionUID = 2459373647785530657L;

	private static AtomicLong indexOfNextRandomVariable = new AtomicLong(0);

	private static enum OperatorType {
		ADD, MULT, DIV, SUB, SQUARED, SQRT, LOG, SIN, COS, EXP, INVERT, CAP, FLOOR, ABS, 
		ADDPRODUCT, ADDRATIO, SUBRATIO, BARRIER, DISCOUNT, ACCURUE, POW, MIN, MAX, AVERAGE, VARIANCE, 
		STDEV, STDERROR, SVARIANCE, AVERAGE2, VARIANCE2, 
		STDEV2, STDERROR2
	}

	private static class OperatorTreeNode {
		private final Long id;
		private final OperatorType operator;
		private final List<OperatorTreeNode> arguments;
		private final List<RandomVariableInterface> argumentValues;

		public OperatorTreeNode(OperatorType operator, List<RandomVariableInterface> arguments) {
			this(operator,
					arguments != null ? arguments.stream().map((RandomVariableInterface x) -> {
						return (x != null && x instanceof RandomVariableDifferentiableAAD2) ? ((RandomVariableDifferentiableAAD2)x).getOperatorTreeNode(): null;
					}).collect(Collectors.toList()) : null,
							arguments != null ? arguments.stream().map((RandomVariableInterface x) -> {
						return (x != null && x instanceof RandomVariableDifferentiableAAD2) ? ((RandomVariableDifferentiableAAD2)x).values : x;
					}).collect(Collectors.toList()) : null
					);

		}
		public OperatorTreeNode(OperatorType operator, List<OperatorTreeNode> arguments, List<RandomVariableInterface> argumentValues) {
			super();
			this.id = indexOfNextRandomVariable.getAndIncrement();
			this.operator = operator;
			this.arguments = arguments;
			this.argumentValues = (operator != null && operator.equals(OperatorType.ADD)) ? null: argumentValues;
		}
		
		private void propagateDerivativesFromResultToArgument(Map<Long, RandomVariableInterface> derivatives) {

			for(OperatorTreeNode argument : arguments) {
				if(argument != null) {
					Long argumentID = argument.id;
					if(!derivatives.containsKey(argumentID)) derivatives.put(argumentID, new RandomVariable(0.0));

					RandomVariableInterface partialDerivative	= getPartialDerivative(argument);
					RandomVariableInterface derivative			= derivatives.get(id);
					RandomVariableInterface argumentDerivative	= derivatives.get(argumentID);

					argumentDerivative = argumentDerivative.addProduct(partialDerivative, derivative);

					derivatives.put(argumentID, argumentDerivative);
				}
			}
		}

		private RandomVariableInterface getPartialDerivative(OperatorTreeNode differential){

			if(!arguments.contains(differential)) return new RandomVariable(0.0);

			int differentialIndex = arguments.indexOf(differential);
			RandomVariableInterface X = arguments.size() > 0 && argumentValues != null ? argumentValues.get(0) : null;
			RandomVariableInterface Y = arguments.size() > 1 && argumentValues != null ? argumentValues.get(1) : null;
			RandomVariableInterface Z = arguments.size() > 2 && argumentValues != null ? argumentValues.get(2) : null;

			RandomVariableInterface resultrandomvariable = null;

			switch(operator) {
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
				resultrandomvariable = new RandomVariable(X.size()).invert();
				break;
			case VARIANCE:
				resultrandomvariable = X.sub(X.getAverage()*(2.0*X.size()-1.0)/X.size()).mult(2.0/X.size());
				break;
			case STDEV:
				resultrandomvariable = X.sub(X.getAverage()*(2.0*X.size()-1.0)/X.size()).mult(2.0/X.size()).mult(0.5).div(Math.sqrt(X.getVariance()));
				break;
			case MIN:
				double min = X.getMin();
				resultrandomvariable = X.apply(x -> (x == min) ? 1.0 : 0.0);
				break;
			case MAX:
				double max = X.getMax();
				resultrandomvariable = X.apply(x -> (x == max) ? 1.0 : 0.0);
				break;
			case ABS:
				resultrandomvariable = X.barrier(X, new RandomVariable(1.0), new RandomVariable(-1.0));
				break;
			case STDERROR:
				resultrandomvariable = X.sub(X.getAverage()*(2.0*X.size()-1.0)/X.size()).mult(2.0/X.size()).mult(0.5).div(Math.sqrt(X.getVariance() * X.size()));
				break;
			case SVARIANCE:
				resultrandomvariable = X.sub(X.getAverage()*(2.0*X.size()-1.0)/X.size()).mult(2.0/(X.size()-1));
				break;
			case ADD:
				resultrandomvariable = new RandomVariable(1.0);
				break;
			case SUB:
				resultrandomvariable = new RandomVariable(differentialIndex == 0 ? 1.0 : -1.0);
				break;
			case MULT:
				resultrandomvariable = differentialIndex == 0 ? Y : X;
				break;
			case DIV:
				resultrandomvariable = differentialIndex == 0 ? Y.invert() : X.div(Y.squared());
				break;
			case CAP:
				if(differentialIndex == 0) {
					resultrandomvariable = X.barrier(X.sub(Y), new RandomVariable(0.0), new RandomVariable(1.0));
				}
				else {
					resultrandomvariable = X.barrier(X.sub(Y), new RandomVariable(1.0), new RandomVariable(0.0));
				}
				break;
			case FLOOR:
				if(differentialIndex == 0) {
					resultrandomvariable = X.barrier(X.sub(Y), new RandomVariable(1.0), new RandomVariable(0.0));
				}
				else {
					resultrandomvariable = X.barrier(X.sub(Y), new RandomVariable(0.0), new RandomVariable(1.0));
				}
				break;
			case AVERAGE2:
				resultrandomvariable = differentialIndex == 0 ? Y : X;
				break;
			case VARIANCE2:
				resultrandomvariable = differentialIndex == 0 ? Y.mult(2.0).mult(X.mult(Y.add(X.getAverage(Y)*(X.size()-1)).sub(X.getAverage(Y)))) :
					X.mult(2.0).mult(Y.mult(X.add(Y.getAverage(X)*(X.size()-1)).sub(Y.getAverage(X))));
				break;
			case STDEV2:		
				resultrandomvariable = differentialIndex == 0 ? Y.mult(2.0).mult(X.mult(Y.add(X.getAverage(Y)*(X.size()-1)).sub(X.getAverage(Y)))).div(Math.sqrt(X.getVariance(Y))) :
					X.mult(2.0).mult(Y.mult(X.add(Y.getAverage(X)*(X.size()-1)).sub(Y.getAverage(X)))).div(Math.sqrt(Y.getVariance(X)));
				break;
			case STDERROR2:				
				resultrandomvariable = differentialIndex == 0 ? Y.mult(2.0).mult(X.mult(Y.add(X.getAverage(Y)*(X.size()-1)).sub(X.getAverage(Y)))).div(Math.sqrt(X.getVariance(Y) * X.size())) :
					X.mult(2.0).mult(Y.mult(X.add(Y.getAverage(X)*(X.size()-1)).sub(Y.getAverage(X)))).div(Math.sqrt(Y.getVariance(X) * Y.size()));
				break;
			case POW:
				/* second argument will always be deterministic and constant! */
				resultrandomvariable = (differentialIndex == 0) ? Y.mult(X.pow(Y.getAverage() - 1.0)) : new RandomVariable(0.0);
			case ADDPRODUCT:
				if(differentialIndex == 0) {
					resultrandomvariable = new RandomVariable(1.0);
				} else if(differentialIndex == 1) {
					resultrandomvariable = Z;
				} else {
					resultrandomvariable = Y;
				}
				break;
			case ADDRATIO:
				if(differentialIndex == 0) {
					resultrandomvariable = new RandomVariable(1.0);
				} else if(differentialIndex == 1) {
					resultrandomvariable = Z.invert();
				} else {
					resultrandomvariable = Y.div(Z.squared());
				}
				break;
			case SUBRATIO:
				if(differentialIndex == 0) {
					resultrandomvariable = new RandomVariable(1.0);
				} else if(differentialIndex == 1) {
					resultrandomvariable = Z.invert().mult(-1.0);
				} else {
					resultrandomvariable = Y.div(Z.squared()).mult(-1.0);
				}
				break;
			case ACCURUE:
				if(differentialIndex == 0) {
					resultrandomvariable = Y.mult(Z).add(1.0);
				} else if(differentialIndex == 1) {
					resultrandomvariable = X.mult(Z);
				} else {
					resultrandomvariable = X.mult(Y);
				}
				break;
			case DISCOUNT:
				if(differentialIndex == 0) {
					resultrandomvariable = Y.mult(Z).add(1.0).invert();
				} else if(differentialIndex == 1) {
					resultrandomvariable = X.mult(Z).div(Y.mult(Z).add(1.0).squared());
				} else {
					resultrandomvariable = X.mult(Y).div(Y.mult(Z).add(1.0).squared());
				}
				break;
			case BARRIER:
				if(differentialIndex == 0) {
					resultrandomvariable = X.apply(x -> (x == 0.0) ? Double.POSITIVE_INFINITY : 0.0);
				} else if(differentialIndex == 1) {
					resultrandomvariable = X.barrier(X, new RandomVariable(1.0), new RandomVariable(0.0));
				} else {
					resultrandomvariable = X.barrier(X, new RandomVariable(0.0), new RandomVariable(1.0));
				}
			default:
				break;
			}

			return resultrandomvariable;
		}
	}

	private final RandomVariableInterface values;
	private final OperatorTreeNode operatorTreeNode;

	public static RandomVariableDifferentiableAAD2 of(double value) {
		return new RandomVariableDifferentiableAAD2(value);
	}

	public static RandomVariableDifferentiableAAD2 of(RandomVariableInterface randomVariable) {
		return new RandomVariableDifferentiableAAD2(randomVariable);
	}

	public RandomVariableDifferentiableAAD2(double value) {
		this(new RandomVariable(value), null, null);
	}

	public RandomVariableDifferentiableAAD2(double time, double[] realisations) {
		this(new RandomVariable(time, realisations), null, null);
	}

	public RandomVariableDifferentiableAAD2(RandomVariableInterface randomVariable) {
		this(randomVariable, null, null);
	}

	private RandomVariableDifferentiableAAD2(RandomVariableInterface values, List<RandomVariableInterface> arguments, OperatorType operator) {
		super();
		this.values = values;
		this.operatorTreeNode = new OperatorTreeNode(operator, arguments);
	}

	public RandomVariableInterface getRandomVariable() {
		return values;
	}

	public OperatorTreeNode getOperatorTreeNode() {
		return operatorTreeNode;
	}

	private RandomVariableInterface apply(OperatorType operator, RandomVariableInterface[] randomVariableInterfaces){

		// Construct argument list
		ArrayList<RandomVariableInterface> arguments = new ArrayList<RandomVariableInterface>();
		for(RandomVariableInterface randomVariable : randomVariableInterfaces) {
			arguments.add(randomVariable);
		}

		// Calculate values
		RandomVariableInterface result;
		RandomVariableInterface X = null,Y = null,Z = null;

		if(arguments.size() > 0) X = valuesOf(arguments.get(0));
		if(arguments.size() > 1) Y = valuesOf(arguments.get(1));
		if(arguments.size() > 2) Z = valuesOf(arguments.get(2));

		switch(operator){
		case SQUARED:
			result = X.squared();
			break;
		case SQRT:
			result = X.sqrt();
			break;
		case EXP:
			result = X.exp();
			break;
		case LOG:
			result = X.log();
			break;
		case SIN:
			result = X.sin();
			break;
		case COS:
			result = X.cos();
			break;
		case ABS:
			result = X.abs();
			break;
		case INVERT:
			result = X.invert();
			break;
		case AVERAGE:
			result = new RandomVariable(X.getAverage());
			break;
		case STDERROR:
			result = new RandomVariable(X.getStandardError());
			break;
		case STDEV:
			result = new RandomVariable(X.getStandardDeviation());
			break;
		case VARIANCE:
			result = new RandomVariable(X.getVariance());
			break;
		case SVARIANCE:
			result = new RandomVariable(X.getSampleVariance());
			break;
		case ADD:
			result = X.add(Y);
			break;
		case SUB:
			result = X.sub(Y);
			break;
		case MULT:
			result = X.mult(Y);
			break;
		case DIV:
			result = X.div(Y);
			break;
		case CAP:
			result = X.cap(Y);
			break;
		case FLOOR:
			result = X.floor(Y);
			break;			
		case POW:
			result = X.pow( /* argument is deterministic random variable */ Y.getAverage());
			break;
		case AVERAGE2:
			result = new RandomVariable(X.getAverage(Y));
			break;
		case STDERROR2:
			result = new RandomVariable(X.getStandardError(Y));
			break;
		case STDEV2:
			result = new RandomVariable(X.getStandardDeviation(Y));
			break;
		case VARIANCE2:
			result = new RandomVariable(X.getVariance(Y));
			break;
		case ADDPRODUCT:
			result = X.addProduct(Y,Z);
			break;
		case ADDRATIO:
			result = X.addRatio(Y, Z);
			break;
		case SUBRATIO:
			result = X.subRatio(Y, Z);
			break;
		case ACCURUE:
			result = X.accrue(Y, /* second argument is deterministic anyway */ Z.getAverage());
			break;
		case DISCOUNT:
			result = X.discount(Y, /* second argument is deterministic anyway */ Z.getAverage());
			break;
		default:
			throw new IllegalArgumentException("Operation not supported!\n");
		}

		/* create new RandomVariableUniqueVariable which is definitely NOT Constant */
		RandomVariableDifferentiableAAD2 newRandomVariableAAD = new RandomVariableDifferentiableAAD2(result, arguments, operator);

		/* return new RandomVariable */
		return newRandomVariableAAD;
	}


	public Long getID(){
		return getOperatorTreeNode().id;
	}

	public Map<Long, RandomVariableInterface> getGradient() {

		// The map maintaining the derivatives id -> derivative
		Map<Long, RandomVariableInterface> derivatives = new HashMap<Long, RandomVariableInterface>();

		// Put derivative of this node w.r.t. itself
		derivatives.put(getID(), new RandomVariable(1.0));

		// The set maintaining the independents. Note: TreeMap is maintaining a sort on the keys.
		TreeMap<Long, OperatorTreeNode> independents = new TreeMap<Long, OperatorTreeNode>();
		independents.put(getID(), getOperatorTreeNode());

		while(independents.size() > 0) {
			// Process node with the highest id in independents
			Map.Entry<Long, OperatorTreeNode> independentEntry = independents.lastEntry();
			Long id = independentEntry.getKey();
			OperatorTreeNode independent = independentEntry.getValue();

			// Get arguments of this node and propagate derivative to arguments
			List<OperatorTreeNode> arguments = independent.arguments;
			if(arguments != null && arguments.size() > 0) {
				independent.propagateDerivativesFromResultToArgument(derivatives);

				// Add all non constant arguments to the list of independents
				for(OperatorTreeNode argument : arguments) {
					if(argument != null) {
						Long argumentId = argument.id;
						independents.put(argumentId, argument);
					}
				}

				// Remove id from derivatives - keep only leaf nodes.
				derivatives.remove(id);
			}

			// Done with processing. Remove from map.
			independents.remove(id);
		}

		return derivatives;
	}

	/* for all functions that need to be differentiated and are returned as double in the Interface, write a method to return it as RandomVariableAAD 
	 * that is deterministic by its nature. For their double-returning pendant just return the average of the deterministic RandomVariableAAD  */

	public RandomVariableInterface average(RandomVariableInterface probabilities) {
		/*returns deterministic AAD random variable */
		return apply(OperatorType.AVERAGE2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface variance(RandomVariableInterface probabilities){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.VARIANCE2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface 	standardDeviation(RandomVariableInterface probabilities){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.STDEV2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface 	standardError(RandomVariableInterface probabilities){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.STDERROR2, new RandomVariableInterface[]{this, probabilities});
	}

	public RandomVariableInterface average(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.AVERAGE, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface variance(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.VARIANCE, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface sampleVariance() {
		/*returns deterministic AAD random variable */
		return apply(OperatorType.SVARIANCE, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface 	standardDeviation(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.STDEV, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface standardError(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.STDERROR, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface 	min(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.MIN, new RandomVariableInterface[]{this});
	}

	public RandomVariableInterface 	max(){
		/*returns deterministic AAD random variable */
		return apply(OperatorType.MAX, new RandomVariableInterface[]{this});
	}

	/* setter and getter */

	private static RandomVariableInterface valuesOf(RandomVariableInterface randomVariable){
		return randomVariable instanceof RandomVariableDifferentiableAAD2 ? 
				((RandomVariableDifferentiableAAD2) randomVariable).values : randomVariable;
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
		return valuesOf(this).getMin();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getMax()
	 */
	@Override
	public double getMax() {
		return valuesOf(this).getMax();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getAverage()
	 */
	@Override
	public double getAverage() {
		return valuesOf(this).getAverage();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getAverage(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getAverage(RandomVariableInterface probabilities) {
		return valuesOf(this).getAverage(probabilities);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getVariance()
	 */
	@Override
	public double getVariance() {
		return valuesOf(this).getVariance();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getVariance(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getVariance(RandomVariableInterface probabilities) {
		return valuesOf(this).getVariance(probabilities);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getSampleVariance()
	 */
	@Override
	public double getSampleVariance() {
		return valuesOf(this).getSampleVariance();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardDeviation()
	 */
	@Override
	public double getStandardDeviation() {
		return valuesOf(this).getStandardDeviation();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardDeviation(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getStandardDeviation(RandomVariableInterface probabilities) {
		return valuesOf(this).getStandardDeviation(probabilities);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardError()
	 */
	@Override
	public double getStandardError() {
		return valuesOf(this).getStandardError();
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getStandardError(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getStandardError(RandomVariableInterface probabilities) {
		return valuesOf(this).getStandardError(probabilities);
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
		return ((RandomVariableDifferentiableAAD2) valuesOf(this)).valuesOf(this).getQuantile(quantile, probabilities);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getQuantileExpectation(double, double)
	 */
	@Override
	public double getQuantileExpectation(double quantileStart, double quantileEnd) {
		return ((RandomVariableDifferentiableAAD2) valuesOf(this)).valuesOf(this).getQuantileExpectation(quantileStart, quantileEnd);
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
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).cap(cap),
				Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(cap) }),
				OperatorType.CAP);
	}

	@Override
	public RandomVariableInterface floor(double floor) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).floor(floor),
				Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(floor) }),
				OperatorType.FLOOR);
	}

	@Override
	public RandomVariableInterface add(double value) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).add(value),
				Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(value) }),
				OperatorType.ADD);
	}

	@Override
	public RandomVariableInterface sub(double value) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).sub(value),
				Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(value) }),
				OperatorType.SUB);
	}

	@Override
	public RandomVariableInterface mult(double value) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).mult(value),
				Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(value) }),
				OperatorType.MULT);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#div(double)
	 */
	@Override
	public RandomVariableInterface div(double value) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).div(value),
				Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(value) }),
				OperatorType.DIV);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#pow(double)
	 */
	@Override
	public RandomVariableInterface pow(double exponent) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).pow(exponent),
				Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(exponent) }),
				OperatorType.POW);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#squared()
	 */
	@Override
	public RandomVariableInterface squared() {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).squared(),
				Arrays.asList(new RandomVariableInterface[]{ this }),
				OperatorType.SQUARED);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sqrt()
	 */
	@Override
	public RandomVariableInterface sqrt() {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).sqrt(),
				Arrays.asList(new RandomVariableInterface[]{ this }),
				OperatorType.SQRT);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#exp()
	 */
	@Override
	public RandomVariableInterface exp() {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).exp(),
				Arrays.asList(new RandomVariableInterface[]{ this }),
				OperatorType.EXP);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#log()
	 */
	@Override
	public RandomVariableInterface log() {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).log(),
				Arrays.asList(new RandomVariableInterface[]{ this }),
				OperatorType.LOG);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sin()
	 */
	@Override
	public RandomVariableInterface sin() {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).sin(),
				Arrays.asList(new RandomVariableInterface[]{ this }),
				OperatorType.SIN);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#cos()
	 */
	@Override
	public RandomVariableInterface cos() {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).cos(),
				Arrays.asList(new RandomVariableInterface[]{ this }),
				OperatorType.COS);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#add(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface add(RandomVariableInterface randomVariable) {	
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).add(randomVariable),
				Arrays.asList(new RandomVariableInterface[]{ this, randomVariable }),
				OperatorType.ADD);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sub(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface sub(RandomVariableInterface randomVariable) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).sub(randomVariable),
				Arrays.asList(new RandomVariableInterface[]{ this, randomVariable }),
				OperatorType.SUB);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#mult(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableDifferentiableInterface mult(RandomVariableInterface randomVariable) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).mult(randomVariable),
				Arrays.asList(new RandomVariableInterface[]{ this, randomVariable }),
				OperatorType.MULT);
	}

	@Override
	public RandomVariableInterface div(RandomVariableInterface randomVariable) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).div(randomVariable),
				Arrays.asList(new RandomVariableInterface[]{ this, randomVariable }),
				OperatorType.DIV);
	}

	@Override
	public RandomVariableInterface cap(RandomVariableInterface cap) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).cap(cap),
				Arrays.asList(new RandomVariableInterface[]{ this, cap }),
				OperatorType.CAP);
	}

	@Override
	public RandomVariableInterface floor(RandomVariableInterface floor) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).cap(floor),
				Arrays.asList(new RandomVariableInterface[]{ this, floor }),
				OperatorType.FLOOR);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#accrue(net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface accrue(RandomVariableInterface rate, double periodLength) {
		return apply(OperatorType.ACCURUE, new RandomVariableInterface[]{this, rate, new RandomVariable(periodLength)});
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#discount(net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface discount(RandomVariableInterface rate, double periodLength) {
		return apply(OperatorType.DISCOUNT, new RandomVariableInterface[]{this, rate, new RandomVariable(periodLength)});
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
		return apply(OperatorType.BARRIER, new RandomVariableInterface[]{this, valueIfTriggerNonNegative, new RandomVariable(valueIfTriggerNegative)});
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
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).abs(),
				Arrays.asList(new RandomVariableInterface[]{ this }),
				OperatorType.ABS);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addProduct(net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface addProduct(RandomVariableInterface factor1, double factor2) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).addProduct(factor1, factor2),
				Arrays.asList(new RandomVariableInterface[]{ this, factor1, new RandomVariable(factor2) }),
				OperatorType.ADDPRODUCT);
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addProduct(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface addProduct(RandomVariableInterface factor1, RandomVariableInterface factor2) {
		return new RandomVariableDifferentiableAAD2(
				valuesOf(this).addProduct(factor1, factor2),
				Arrays.asList(new RandomVariableInterface[]{ this, factor1, factor2 }),
				OperatorType.ADDPRODUCT);
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
