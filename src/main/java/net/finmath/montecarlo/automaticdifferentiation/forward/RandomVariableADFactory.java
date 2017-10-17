/**
 * 
 */
package net.finmath.montecarlo.automaticdifferentiation.forward;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import net.finmath.functions.DoubleTernaryOperator;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.RandomVariable;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.stochastic.ConditionalExpectationEstimatorInterface;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * 
 * @author Stefan Sedlmair
 *
 */
public class RandomVariableADFactory extends AbstractRandomVariableDifferentiableFactory {

	private static AtomicLong indexOfNextRandomVariable = new AtomicLong(0);
	
	private static enum OperatorType {
		ADD, MULT, DIV, SUB, SQUARED, SQRT, LOG, SIN, COS, EXP, INVERT, CAP, FLOOR, ABS, 
		ADDPRODUCT, ADDRATIO, SUBRATIO, BARRIER, DISCOUNT, ACCRUE, POW, MIN, MAX, AVERAGE, VARIANCE, 
		STDEV, STDERROR, SVARIANCE, AVERAGE2, VARIANCE2, 
		STDEV2, STDERROR2, CONDITIONAL_EXPECTATION
	}
		
	private final boolean keepAllDerivativesOfOperatorTree;
	private final double barrierDiracWidth;
		
	/**
	 * @param randomVariableFactoryForNonDifferentiable
	 */
	public RandomVariableADFactory(AbstractRandomVariableFactory randomVariableFactoryForNonDifferentiable, Map<String, Object> properites) {
		super(randomVariableFactoryForNonDifferentiable);
		
		this.barrierDiracWidth 					= (double) properites.getOrDefault("barrierDiracWidth", 1E-4);
		this.keepAllDerivativesOfOperatorTree 	= (boolean) properites.getOrDefault("keepAllDerivativesOfOperatorTree", false);
	}
	
	public RandomVariableADFactory(AbstractRandomVariableFactory randomVariableFactoryForNonDifferentiable){
		this(randomVariableFactoryForNonDifferentiable, new HashMap<>());
	}
	
	
	public RandomVariableADFactory(){
		this(new RandomVariableFactory());
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableFactory#createRandomVariable(double, double)
	 */
	@Override
	public RandomVariableDifferentiableInterface createRandomVariable(double time, double value) {
		// create the value part
		RandomVariableInterface valuePart = super.createRandomVariableNonDifferentiable(time, value);

		return createRandomVariableAD(valuePart, null, null, null, this);
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableFactory#createRandomVariable(double, double[])
	 */
	@Override
	public RandomVariableDifferentiableInterface createRandomVariable(double time, double[] values) {
		// create the value part
		RandomVariableInterface valuePart = super.createRandomVariableNonDifferentiable(time, values);
		
		return createRandomVariableAD(valuePart, null, null, null, this);
	}
	
	protected RandomVariableAD createRandomVariableAD(RandomVariableInterface values, List<RandomVariableInterface> parents, ConditionalExpectationEstimatorInterface estimator,
			OperatorType operator, RandomVariableADFactory factory){
		
		return new RandomVariableAD(values, parents, estimator, operator, factory);
	}
	
	protected RandomVariableAD createRandomVariableAD(RandomVariableInterface values, List<RandomVariableInterface> parents,
			OperatorType operator, RandomVariableADFactory factory){
		return createRandomVariableAD(values, parents, null /*estimator*/, operator, factory);
	}

	public double getBarrierDiracWidth() {
		return barrierDiracWidth;
	}
		
	public boolean keepAllDerivativesOfOperatorTree(){
		return keepAllDerivativesOfOperatorTree;
	}
	
	public static class RandomVariableAD implements RandomVariableDifferentiableInterface{
		
		private static final long serialVersionUID = 2213312413736379638L;
		
		/*
		 * Data model. We maintain the underlying values and a link to the node in the operator tree.
		 */
		private RandomVariableInterface values;
		private final OperatorTreeNode operatorTreeNode;
		private final RandomVariableADFactory factory;

//		public static RandomVariableAD of(double value) {
//			return new RandomVariableAD(value);
//		}
//
//		public static RandomVariableAD of(RandomVariableInterface randomVariable) {
//			return new RandomVariableAD(randomVariable);
//		}

//		public RandomVariableAD(double value) {
//			this(new RandomVariable(value), null, null, null);
//		}
//
//		public RandomVariableAD(RandomVariableInterface randomVariable) {
//			this(randomVariable, null, null, null);
//		}
//
//		public RandomVariableAD(RandomVariableInterface values, RandomVariableADFactory factory) {
//			this(values, null, null, factory);
//		}

		private RandomVariableAD(RandomVariableInterface values, List<RandomVariableInterface> parents, ConditionalExpectationEstimatorInterface estimator,
				OperatorType operator, RandomVariableADFactory factory) {
			super();
			this.values = values;
			this.operatorTreeNode = new OperatorTreeNode(operator,parents, estimator, factory);
			this.factory = factory;
		}

		public OperatorTreeNode getOperatorTreeNode() {
			return operatorTreeNode;
		}

		/**
		 * Returns the underlying values.
		 * 
		 * @return The underling values.
		 */
		private RandomVariableInterface getValues(){
			return values;
		}

		public RandomVariableADFactory getFactory() {
			return factory;
		}

		public Long getID(){
			return getOperatorTreeNode().id;
		}
		
		
		
		@Override
		/**
		 * VERY INEFFICIENT!
		 * 
		 * This RandomVariable implements the AD Algorithm and thus calculates a reverse gradient
		 * 
		 * */
		public Map<Long, RandomVariableInterface> getGradient() {
			Map<Long, RandomVariableInterface> gradient = new HashMap<>();
			
			for(Long leafID : operatorTreeNode.leafNodes.keySet()){
				Map<Long, RandomVariableInterface> reversGradientWRTLeaf = getAllPartialDerivativesFor(leafID);
				gradient.put(leafID, reversGradientWRTLeaf.get(getID()));
			}
			
			return gradient;
		}
		
		/**
		 * The method implements an AD algorithm, scaling with the number of inputs.
		 * 
		 * Returning map includes all identifiers of (final) values that are dependent on the parameter (stated in argument), 
		 * and their partial derivatives with respect to this parameter
		 * 
		 * @param leafID identifier of the parameter to calculate the derivatives
		 * 
		 * @return map with key: identifiers of (final) values and the values of the partial derivative with respect to the leafID 
		 * 
		 * 
		 * @author Stefan Sedlmair
		 * @author Christian Fries
		 * */
		public Map<Long, RandomVariableInterface> getAllPartialDerivativesFor(Long leafID) {
			return this.operatorTreeNode.leafNodes.get(leafID).getAllPartialDerivatives();
		}
		
		public Map<Long, RandomVariableInterface> getAllPartialDerivatives() {
			return this.operatorTreeNode.getAllPartialDerivatives();
		} 
		
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
		 * @see net.finmath.stochastic.RandomVariableInterface#getMin()
		 */
		@Override
		public double getMin() {
			return getValues().getMin();
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getMax()
		 */
		@Override
		public double getMax() {
			return getValues().getMax();
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getAverage()
		 */
		@Override
		public double getAverage() {
			return getValues().getAverage();
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getAverage(net.finmath.stochastic.RandomVariableInterface)
		 */
		@Override
		public double getAverage(RandomVariableInterface probabilities) {
			return getValues().getAverage(probabilities);
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getVariance()
		 */
		@Override
		public double getVariance() {
			return getValues().getVariance();
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getVariance(net.finmath.stochastic.RandomVariableInterface)
		 */
		@Override
		public double getVariance(RandomVariableInterface probabilities) {
			return getValues().getVariance(probabilities);
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getSampleVariance()
		 */
		@Override
		public double getSampleVariance() {
			return getValues().getSampleVariance();
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getStandardDeviation()
		 */
		@Override
		public double getStandardDeviation() {
			return getValues().getStandardDeviation();
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getStandardDeviation(net.finmath.stochastic.RandomVariableInterface)
		 */
		@Override
		public double getStandardDeviation(RandomVariableInterface probabilities) {
			return getValues().getStandardDeviation(probabilities);
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getStandardError()
		 */
		@Override
		public double getStandardError() {
			return getValues().getStandardError();
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getStandardError(net.finmath.stochastic.RandomVariableInterface)
		 */
		@Override
		public double getStandardError(RandomVariableInterface probabilities) {
			return getValues().getStandardError(probabilities);
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
			return ((RandomVariableAD) getValues()).getValues().getQuantile(quantile, probabilities);
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getQuantileExpectation(double, double)
		 */
		@Override
		public double getQuantileExpectation(double quantileStart, double quantileEnd) {
			return ((RandomVariableAD) getValues()).getValues().getQuantileExpectation(quantileStart, quantileEnd);
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

		/*
		 * The following methods are operations with are differntiable.
		 */

		@Override
		public RandomVariableInterface cache() {
			values = values.cache();
			return this;
		}

		@Override
		public RandomVariableInterface cap(double cap) {
			return getFactory().createRandomVariableAD(
					getValues().cap(cap),
					Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(cap) }),
					OperatorType.CAP,
					getFactory());
		}

		@Override
		public RandomVariableInterface floor(double floor) {
			return getFactory().createRandomVariableAD(
					getValues().floor(floor),
					Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(floor) }),
					OperatorType.FLOOR,
					getFactory());
		}

		@Override
		public RandomVariableInterface add(double value) {
			return getFactory().createRandomVariableAD(
					getValues().add(value),
					Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(value) }),
					OperatorType.ADD,
					getFactory());
		}

		@Override
		public RandomVariableInterface sub(double value) {
			return getFactory().createRandomVariableAD(
					getValues().sub(value),
					Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(value) }),
					OperatorType.SUB,
					getFactory());
		}

		@Override
		public RandomVariableInterface mult(double value) {
			return getFactory().createRandomVariableAD(
					getValues().mult(value),
					Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(value) }),
					OperatorType.MULT,
					getFactory());
		}

		@Override
		public RandomVariableInterface div(double value) {
			return getFactory().createRandomVariableAD(
					getValues().div(value),
					Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(value) }),
					OperatorType.DIV,
					getFactory());
		}

		@Override
		public RandomVariableInterface pow(double exponent) {
			return getFactory().createRandomVariableAD(
					getValues().pow(exponent),
					Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(exponent) }),
					OperatorType.POW,
					getFactory());
		}

		@Override
		public RandomVariableInterface average() {
			return getFactory().createRandomVariableAD(
					getValues().average(),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.AVERAGE,
					getFactory());
		}

		public RandomVariableInterface getConditionalExpectation(ConditionalExpectationEstimatorInterface estimator) {
			return getFactory().createRandomVariableAD(
					getValues().getConditionalExpectation(estimator),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					estimator,
					OperatorType.CONDITIONAL_EXPECTATION,
					getFactory());

		}

		@Override
		public RandomVariableInterface squared() {
			return getFactory().createRandomVariableAD(
					getValues().squared(),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.SQUARED,
					getFactory());
		}

		@Override
		public RandomVariableInterface sqrt() {
			return getFactory().createRandomVariableAD(
					getValues().sqrt(),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.SQRT,
					getFactory());
		}

		@Override
		public RandomVariableInterface exp() {
			return getFactory().createRandomVariableAD(
					getValues().exp(),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.EXP,
					getFactory());
		}

		@Override
		public RandomVariableInterface log() {
			return getFactory().createRandomVariableAD(
					getValues().log(),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.LOG,
					getFactory());
		}

		@Override
		public RandomVariableInterface sin() {
			return getFactory().createRandomVariableAD(
					getValues().sin(),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.SIN,
					getFactory());
		}

		@Override
		public RandomVariableInterface cos() {
			return getFactory().createRandomVariableAD(
					getValues().cos(),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.COS,
					getFactory());
		}

		@Override
		public RandomVariableInterface add(RandomVariableInterface randomVariable) {	
			return getFactory().createRandomVariableAD(
					getValues().add(randomVariable),
					Arrays.asList(new RandomVariableInterface[]{ this, randomVariable }),
					OperatorType.ADD,
					getFactory());
		}

		@Override
		public RandomVariableInterface sub(RandomVariableInterface randomVariable) {
			return getFactory().createRandomVariableAD(
					getValues().sub(randomVariable),
					Arrays.asList(new RandomVariableInterface[]{ this, randomVariable }),
					OperatorType.SUB,
					getFactory());
		}

		@Override
		public RandomVariableDifferentiableInterface mult(RandomVariableInterface randomVariable) {
			return getFactory().createRandomVariableAD(
					getValues().mult(randomVariable),
					Arrays.asList(new RandomVariableInterface[]{ this, randomVariable }),
					OperatorType.MULT,
					getFactory());
		}

		@Override
		public RandomVariableInterface div(RandomVariableInterface randomVariable) {
			return getFactory().createRandomVariableAD(
					getValues().div(randomVariable),
					Arrays.asList(new RandomVariableInterface[]{ this, randomVariable }),
					OperatorType.DIV,
					getFactory());
		}

		@Override
		public RandomVariableInterface cap(RandomVariableInterface cap) {
			return getFactory().createRandomVariableAD(
					getValues().cap(cap),
					Arrays.asList(new RandomVariableInterface[]{ this, cap }),
					OperatorType.CAP,
					getFactory());
		}

		@Override
		public RandomVariableInterface floor(RandomVariableInterface floor) {
			return getFactory().createRandomVariableAD(
					getValues().cap(floor),
					Arrays.asList(new RandomVariableInterface[]{ this, floor }),
					OperatorType.FLOOR,
					getFactory());
		}

		@Override
		public RandomVariableInterface accrue(RandomVariableInterface rate, double periodLength) {
			return getFactory().createRandomVariableAD(
					getValues().accrue(rate, periodLength),
					Arrays.asList(new RandomVariableInterface[]{ this, rate, new RandomVariable(periodLength) }),
					OperatorType.ACCRUE,
					getFactory());
		}

		@Override
		public RandomVariableInterface discount(RandomVariableInterface rate, double periodLength) {
			return getFactory().createRandomVariableAD(
					getValues().discount(rate, periodLength),
					Arrays.asList(new RandomVariableInterface[]{ this, rate, new RandomVariable(periodLength) }),
					OperatorType.DISCOUNT,
					getFactory());
		}

		@Override
		public RandomVariableInterface barrier(RandomVariableInterface trigger, RandomVariableInterface valueIfTriggerNonNegative, RandomVariableInterface valueIfTriggerNegative) {
			RandomVariableInterface triggerValues = trigger instanceof RandomVariableAD ? ((RandomVariableAD)trigger).getValues() : trigger;
			return getFactory().createRandomVariableAD(
					getValues().barrier(triggerValues, valueIfTriggerNonNegative, valueIfTriggerNegative),
					Arrays.asList(new RandomVariableInterface[]{ trigger, valueIfTriggerNonNegative, valueIfTriggerNegative }),
					OperatorType.BARRIER,
					getFactory());
		}

		@Override
		public RandomVariableInterface barrier(RandomVariableInterface trigger, RandomVariableInterface valueIfTriggerNonNegative, double valueIfTriggerNegative) {
			RandomVariableInterface triggerValues = trigger instanceof RandomVariableAD ? ((RandomVariableAD)trigger).getValues() : trigger;
			return getFactory().createRandomVariableAD(
					getValues().barrier(triggerValues, valueIfTriggerNonNegative, valueIfTriggerNegative),
					Arrays.asList(new RandomVariableInterface[]{ trigger, valueIfTriggerNonNegative, new RandomVariable(valueIfTriggerNegative) }),
					OperatorType.BARRIER,
					getFactory());
		}

		@Override
		public RandomVariableInterface invert() {
			return getFactory().createRandomVariableAD(
					getValues().invert(),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.INVERT,
					getFactory());
		}

		@Override
		public RandomVariableInterface abs() {
			return getFactory().createRandomVariableAD(
					getValues().abs(),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.ABS,
					getFactory());
		}

		@Override
		public RandomVariableInterface addProduct(RandomVariableInterface factor1, double factor2) {
			return getFactory().createRandomVariableAD(
					getValues().addProduct(factor1, factor2),
					Arrays.asList(new RandomVariableInterface[]{ this, factor1, new RandomVariable(factor2) }),
					OperatorType.ADDPRODUCT,
					getFactory());
		}

		@Override
		public RandomVariableInterface addProduct(RandomVariableInterface factor1, RandomVariableInterface factor2) {
			return getFactory().createRandomVariableAD(
					getValues().addProduct(factor1, factor2),
					Arrays.asList(new RandomVariableInterface[]{ this, factor1, factor2 }),
					OperatorType.ADDPRODUCT,
					getFactory());
		}

		@Override
		public RandomVariableInterface addRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
			return getFactory().createRandomVariableAD(
					getValues().addRatio(numerator, denominator),
					Arrays.asList(new RandomVariableInterface[]{ this, numerator, denominator }),
					OperatorType.ADDRATIO,
					getFactory());
		}

		@Override
		public RandomVariableInterface subRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
			return getFactory().createRandomVariableAD(
					getValues().subRatio(numerator, denominator),
					Arrays.asList(new RandomVariableInterface[]{ this, numerator, denominator }),
					OperatorType.SUBRATIO,
					getFactory());
		}

		/*
		 * The following methods are end points, the result is not differentiable.
		 */

		@Override
		public RandomVariableInterface isNaN() {
			return getValues().isNaN();
		}

		@Override
		public IntToDoubleFunction getOperator() {
			return getValues().getOperator();
		}

		@Override
		public DoubleStream getRealizationsStream() {
			return getValues().getRealizationsStream();
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

		/*
		 * The following methods are experimental - will be removed
		 */

		private RandomVariableInterface getAverageAsRandomVariableAD(RandomVariableInterface probabilities) {
			/*returns deterministic AD random variable */
			return getFactory().createRandomVariableAD(
					new RandomVariable(getAverage(probabilities)),
					Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(probabilities) }),
					OperatorType.AVERAGE2,
					getFactory());
		}

		private RandomVariableInterface getVarianceAsRandomVariableAD(RandomVariableInterface probabilities){
			/*returns deterministic AD random variable */
			return getFactory().createRandomVariableAD(
					new RandomVariable(getVariance(probabilities)),
					Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(probabilities) }),
					OperatorType.VARIANCE2,
					getFactory());
		}

		private RandomVariableInterface getStandardDeviationAsRandomVariableAD(RandomVariableInterface probabilities){
			/*returns deterministic AD random variable */
			return getFactory().createRandomVariableAD(
					new RandomVariable(getStandardDeviation(probabilities)),
					Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(probabilities) }),
					OperatorType.STDEV2,
					getFactory());
		}

		private RandomVariableInterface getStandardErrorAsRandomVariableAD(RandomVariableInterface probabilities){
			/*returns deterministic AD random variable */
			return getFactory().createRandomVariableAD(
					new RandomVariable(getStandardError(probabilities)),
					Arrays.asList(new RandomVariableInterface[]{ this, new RandomVariable(probabilities) }),
					OperatorType.STDERROR2,
					getFactory());
		}

		public RandomVariableInterface getVarianceAsRandomVariableAD(){
			/*returns deterministic AD random variable */
			return getFactory().createRandomVariableAD(
					new RandomVariable(getVariance()),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.VARIANCE,
					getFactory());
		}

		public RandomVariableInterface getSampleVarianceAsRandomVariableAD() {
			/*returns deterministic AD random variable */
			return getFactory().createRandomVariableAD(
					new RandomVariable(getSampleVariance()),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.SVARIANCE,
					getFactory());
		}

		public RandomVariableInterface 	getStandardDeviationAsRandomVariableAD(){
			/*returns deterministic AD random variable */
			return getFactory().createRandomVariableAD(
					new RandomVariable(getStandardDeviation()),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.STDEV,
					getFactory());
		}

		public RandomVariableInterface getStandardErrorAsRandomVariableAD(){
			/*returns deterministic AD random variable */
			return getFactory().createRandomVariableAD(
					new RandomVariable(getStandardError()),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.STDERROR,
					getFactory());
		}

		public RandomVariableInterface 	getMinAsRandomVariableAD(){
			/*returns deterministic AD random variable */
			return getFactory().createRandomVariableAD(
					new RandomVariable(getMin()),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.MIN,
					getFactory());
		}

		public RandomVariableInterface 	getMaxAsRandomVariableAD(){
			/*returns deterministic AD random variable */
			return getFactory().createRandomVariableAD(
					new RandomVariable(getMax()),
					Arrays.asList(new RandomVariableInterface[]{ this }),
					OperatorType.MAX,
					getFactory());
		}

	}
	
	/**
	 * A node in the <i>operator tree</i>. It
	 * stores an id (the index m), the operator (the function f_m), and theparentTreeNodes.
	 * It also stores reference to the argument values, if required.
	 * 
	 * @author Christian Fries
	 */
	private static class OperatorTreeNode {
		private final Long id;
		private final OperatorType operatorType;
		private final Object operator;
		
		private final List<RandomVariableInterface> parentValues;
		private final List<OperatorTreeNode> parentTreeNodes;

		private final List<OperatorTreeNode> childTreeNodes;
		
		private final Map<Long, OperatorTreeNode> leafNodes;
		
		private final RandomVariableADFactory factory;
		
		public OperatorTreeNode(OperatorType operatorType, List<RandomVariableInterface> parents, Object estimator, RandomVariableADFactory factory) {
			this.id = indexOfNextRandomVariable.getAndIncrement();
			this.operatorType = operatorType;
			
			// split up parents
			if(parents != null){
				this.parentValues = parents.stream().map(
						(RandomVariableInterface x) -> (x instanceof RandomVariableAD) ? ((RandomVariableAD) x).values :  null
								).collect(Collectors.toList());
				
				this.parentTreeNodes = parents.stream().map(
						(RandomVariableInterface x) -> (x instanceof RandomVariableAD) ? ((RandomVariableAD) x).operatorTreeNode :  null
								).collect(Collectors.toList());
			} else {
				this.parentValues = new ArrayList<>(); 
				this.parentTreeNodes = new ArrayList<>();
			}
			
			// children are always empty
			this.childTreeNodes = new ArrayList<>();
			
			// add this to parents as a child
			for(OperatorTreeNode parentTreeNode : parentTreeNodes) if(parentTreeNode != null) parentTreeNode.childTreeNodes.add(this);					
			
			// look for leaves in parents / if non there be a leaf yourself.
			this.leafNodes = new HashMap<>();
			for(OperatorTreeNode parentTreeNode : parentTreeNodes) if(parentTreeNode != null) leafNodes.putAll(parentTreeNode.leafNodes);
			if(leafNodes.isEmpty()) leafNodes.put(id, this);
			
			// factory
			this.factory = factory;
			
			// estimator for conditional expectation  | TODO: is this still valid for AD? 
			this.operator = estimator;	
		}

		private RandomVariableInterface getPartialDerivative(OperatorTreeNode differential){
				
			if(!parentTreeNodes.contains(differential)) return new RandomVariable(0.0);

			int differentialIndex = parentTreeNodes.indexOf(differential);
			RandomVariableInterface X =parentTreeNodes.size() > 0 && parentValues != null ? parentValues.get(0) : null;
			RandomVariableInterface Y =parentTreeNodes.size() > 1 && parentValues != null ? parentValues.get(1) : null;
			RandomVariableInterface Z =parentTreeNodes.size() > 2 && parentValues != null ? parentValues.get(2) : null;

			RandomVariableInterface resultrandomvariable = null;

			switch(operatorType) {
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
				resultrandomvariable = new RandomVariable(1.0);
				break;
			case CONDITIONAL_EXPECTATION:
				resultrandomvariable = new RandomVariable(1.0);
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
				resultrandomvariable = differentialIndex == 0 ? Y.invert() : X.div(Y.squared()).mult(-1);
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
				break;
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
					resultrandomvariable = Y.div(Z.squared()).mult(-1.0);
				}
				break;
			case SUBRATIO:
				if(differentialIndex == 0) {
					resultrandomvariable = new RandomVariable(1.0);
				} else if(differentialIndex == 1) {
					resultrandomvariable = Z.invert().mult(-1.0);
				} else {
					resultrandomvariable = Y.div(Z.squared());
				}
				break;
			case ACCRUE:
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
					resultrandomvariable = X.mult(Z).div(Y.mult(Z).add(1.0).squared()).mult(-1.0);
				} else {
					resultrandomvariable = X.mult(Y).div(Y.mult(Z).add(1.0).squared()).mult(-1.0);
				}
				break;
			case BARRIER:
				if(differentialIndex == 0) {
					/*
					 * Approximation via local finite difference
					 * (see https://ssrn.com/abstract=2995695 for details).
					 */
					resultrandomvariable = Y.sub(Z);
					double epsilon = factory.getBarrierDiracWidth()*X.getStandardDeviation();
					if(epsilon > 0) {
						resultrandomvariable = resultrandomvariable.mult(X.barrier(X.add(epsilon/2), new RandomVariable(1.0), new RandomVariable(0.0)));
						resultrandomvariable = resultrandomvariable.mult(X.barrier(X.sub(epsilon/2), new RandomVariable(0.0), new RandomVariable(1.0)));
						resultrandomvariable = resultrandomvariable.div(epsilon);
					}
					else {
						resultrandomvariable.mult(0.0);
					}
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
		
		public Map<Long, RandomVariableInterface> getAllPartialDerivatives(){
		
			Map<Long, RandomVariableInterface> partialDerivatives = new HashMap<>();
							
			// every child in the operator tree is of the same instance of its parents
			TreeMap<Long, OperatorTreeNode> treeNodesToPropagte = new TreeMap<>();
			RandomVariableInterface zero = factory.createRandomVariableNonDifferentiable(0.0, 0.0);

			// partial derivative with respect to itself
			partialDerivatives.put(id, factory.createRandomVariableNonDifferentiable(0.0, 1.0));
			
			// add id of lowest variable
			for(OperatorTreeNode childTreeNode : this.childTreeNodes){
				treeNodesToPropagte.put(childTreeNode.id, childTreeNode);
				partialDerivatives.put(childTreeNode.id, childTreeNode.getPartialDerivative(this));
			}
			
			if(!this.childTreeNodes.isEmpty() || factory.keepAllDerivativesOfOperatorTree) partialDerivatives.remove(id);
						
			while(!treeNodesToPropagte.isEmpty()){
				
				// get and remove smallest ID from treeNodesToPropagate
				Entry<Long, OperatorTreeNode> lowestEntry = treeNodesToPropagte.pollFirstEntry();
				
				Long parentID = lowestEntry.getKey();
				OperatorTreeNode parentOperatorTreeNode = lowestEntry.getValue();
				
				// \frac{\partial f}{\partial x}
				RandomVariableInterface parentPartialDerivtivWRTLeaf = partialDerivatives.get(parentID);
				
				for(OperatorTreeNode childTreeNode : parentOperatorTreeNode.childTreeNodes){
					
					Long childID = childTreeNode.id;
					
					// \frac{\partial F}{\partial f}
					RandomVariableInterface childPartialDerivativeWRTParent = childTreeNode.getPartialDerivative(parentOperatorTreeNode);
					
					// chain rule - get already existing part of the sum
					RandomVariableInterface existingChainRuleSum = partialDerivatives.getOrDefault(childID, zero);
					
					// 
					RandomVariableInterface chainRuleSum = existingChainRuleSum.addProduct(childPartialDerivativeWRTParent, parentPartialDerivtivWRTLeaf);
					
					// put result back in reverseGradient
					partialDerivatives.put(childID, chainRuleSum);

					// add all children to workSet to propagate derivatives further upwards
					treeNodesToPropagte.put(childTreeNode.id, childTreeNode);
				}
				
				// if not defined otherwise delete parent after derivative has been propagated upwards to children
				// if no children existed leave if it for the results
				if(!parentOperatorTreeNode.childTreeNodes.isEmpty() || factory.keepAllDerivativesOfOperatorTree) partialDerivatives.remove(parentID); 
				
			}
			
			return partialDerivatives;
		}
		
	}
}
