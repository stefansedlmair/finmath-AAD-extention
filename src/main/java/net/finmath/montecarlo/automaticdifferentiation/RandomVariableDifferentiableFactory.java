/**
 * 
 */
package net.finmath.montecarlo.automaticdifferentiation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.stream.DoubleStream;

import org.apache.commons.math3.util.FastMath;

import net.finmath.functions.DoubleTernaryOperator;
import net.finmath.functions.VectorAlgbra;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.RandomVariable;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.stochastic.ConditionalExpectationEstimatorInterface;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * 
 * 
 * @author Stefan Sedlmair
 * @version 1.0
 */
public class RandomVariableDifferentiableFactory extends AbstractRandomVariableDifferentiableFactory {

	private final static AtomicLong nextIdentifier = new AtomicLong(0); 

	private final double barrierDiracWidth;
	private final double finiteDifferencesStepSize;

	private final boolean enableAD;
	private final boolean retainAllTreeNodes;

	/**
	 * @param randomVariableFactoryForNonDifferentiable
	 */
	public RandomVariableDifferentiableFactory(AbstractRandomVariableFactory randomVariableFactoryForNonDifferentiable, Map<String, Object> properties) {
		super(randomVariableFactoryForNonDifferentiable);
			
		this.barrierDiracWidth 			= (double) properties.getOrDefault("barrierDiracWidth", 0.2);
		this.finiteDifferencesStepSize 	= (double) properties.getOrDefault("finiteDifferencesStepSize", 1E-8);

		this.enableAD 				= (boolean) properties.getOrDefault("enableAD", true);
		this.retainAllTreeNodes 	= (boolean) properties.getOrDefault("retainAllTreeNodes", false);
	}

	public RandomVariableDifferentiableFactory(AbstractRandomVariableFactory randomVariableFactoryForNonDifferentiable) {
		this(randomVariableFactoryForNonDifferentiable, new HashMap<>());
	}

	public RandomVariableDifferentiableFactory() {
		this(new RandomVariableFactory(), new HashMap<>());
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableFactory#createRandomVariable(double, double)
	 */
	@Override
	public RandomVariableDifferentiableInterface createRandomVariable(double time, double value) {
		RandomVariableInterface randomvariable = super.createRandomVariableNonDifferentiable(time, value);
		return new RandomVariableDifferentiable(randomvariable, null, this);
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableFactory#createRandomVariable(double, double[])
	 */
	@Override
	public RandomVariableDifferentiableInterface createRandomVariable(double time, double[] values) {		
		RandomVariableInterface randomvariable = super.createRandomVariableNonDifferentiable(time, values);
		return new RandomVariableDifferentiable(randomvariable, null, this);
	}

	
	/**
	 * Operator Tree Nodes essential for propagation in the operator tree.
	 * 
	 * This implementations does not actively manage the decency structure but relies on the give booleans.
	 * 
	 * @author Stefan Sedlmair
	 * @author Christian Fries
	 * @version 1.0
	 * */
	private static class OperatorTreeNode{
		private final long id;
		private final RandomVariableDifferentiableFactory factory;

		private final List<OperatorTreeNode> parentTreeNodes;
		private final List<RandomVariableInterface> parentValues;

		private final List<OperatorTreeNode> childTreeNodes;

		private final List<Object> derivatives;	
		
		/**
		 * Note derivativeWrtParentRandomVariable can hold {@link ExpectationInformation}.
		 * 
		 * @param parentInfromation list of object arrays. Has to be of the following order: <lu><li>ParentRandomVariable</li><li>derivativeWrtParent</li><li>keepValuesOfParent</li></lu>  
		 * @param factory {@link AbstractRandomVariableDifferentiableFactory} to generate random variables
		 * */
		public OperatorTreeNode(List<Object[]> parentInfromation, RandomVariableDifferentiableFactory factory) {
			// get identifier
			this.id = nextIdentifier.getAndIncrement();

			// get factory
			this.factory = factory;

			// initialize values and allocate memory
			int numberOfParents = parentInfromation != null ? parentInfromation.size() : 0;
			this.parentTreeNodes = new ArrayList<>(numberOfParents);
			this.parentValues = new ArrayList<>(numberOfParents);
			this.derivatives = new ArrayList<>(numberOfParents);

			// if parent information null no parents available
			if(parentInfromation != null) {
				// assign values from parentInfromation
				parentInfromation.stream().forEachOrdered(
						item -> {
							parentTreeNodes.add(treeNodeOf((RandomVariableInterface)item[0]));
							derivatives.add(item[1]);
							parentValues.add(((boolean)item[2]) ? RandomVariableDifferentiable.valuesOf((RandomVariableInterface)item[0]) : null);
						});
				// free memory
				parentInfromation = null;
			}

			// only connect to children if AD is enabled
			if(factory.enableAD) {
				this.childTreeNodes = Collections.synchronizedList(new ArrayList<OperatorTreeNode>());
					for(OperatorTreeNode parentTreeNode : this.parentTreeNodes) 
						if(parentTreeNode != null) 
							parentTreeNode.childTreeNodes.add(this);
			}
			else this.childTreeNodes = null;
		}

		/**
		 * calculated the partial derivative with respect to a direct parent from the operator tree
		 * 
		 * @param differential {@link OperatorTreeNode} of direct parent from operator tree node 
		 * @return derivative with respect to differential 
		 * */
		private RandomVariableInterface getPartialDerivativeFor(OperatorTreeNode differential) {

			if(!parentTreeNodes.contains(differential)) 
				return factory.createRandomVariableNonDifferentiable(-Double.MAX_VALUE, 0) ;

			int differentialIndex = parentTreeNodes.indexOf(differential);		
			Object derivativeFunction = derivatives.get(differentialIndex);

			RandomVariableInterface partialDerivative = null;
			switch(parentTreeNodes.size()){
			case 1:
				partialDerivative = RandomVariableDifferentiable.apply((DoubleUnaryOperator) derivativeFunction, parentValues);
				break;
			case 2:
				partialDerivative = RandomVariableDifferentiable.apply((DoubleBinaryOperator) derivativeFunction, parentValues);
				break;
			case 3:
				partialDerivative = RandomVariableDifferentiable.apply((DoubleTernaryOperator) derivativeFunction, parentValues);
				break;
			default:	
				throw new UnsupportedOperationException();
			}

			return partialDerivative;
		}

		/**
		 * method implements reverse mode of algorithmic differentiation
		 * 
		 * @return {@link Map} with key id and value  dV<sub>this</sub>/dx<sub>id</sub>
		 * */
		private Map<Long, RandomVariableInterface> getGradient() {

			RandomVariableInterface zero = factory.createRandomVariableNonDifferentiable(-Double.MAX_VALUE, 0.0);
			RandomVariableInterface one = factory.createRandomVariableNonDifferentiable(-Double.MAX_VALUE, 1.0);
			
			Map<Long, RandomVariableInterface> gradient = new HashMap<>();

			// every child in the operator tree is of the same instance of its parents
			TreeMap<Long, OperatorTreeNode> treeNodesToPropagte = new TreeMap<>();

			// partial derivative with respect to itself
			gradient.put(id, one);

			// add id of this variable to propagate downwards
			treeNodesToPropagte.put(id, this);			
			
			while(!treeNodesToPropagte.isEmpty()){

				// get and remove highest ID from treeNodesToPropagate
				Entry<Long, OperatorTreeNode> highestEntry = treeNodesToPropagte.pollLastEntry();

				Long childID = highestEntry.getKey();
				OperatorTreeNode childTreeNode = highestEntry.getValue();
				
				final List<OperatorTreeNode> parentTreeNodes = childTreeNode.parentTreeNodes;
				for(int i = 0; i < parentTreeNodes.size(); i++) {
					OperatorTreeNode parentTreeNode = parentTreeNodes.get(i);

					// if parentTreeNode is null, derivative is zero, thus continue with next treeNode
					if(parentTreeNode == null) continue;

					// get the current parent id
					final Long parentID = parentTreeNode.id;
	
					// \frac{\partial f_N}{\partial f_n} | has to exist by construction!
					RandomVariableInterface originPartialDerivtivWRTchild = gradient.get(childID);

					// \frac{\partial f_n}{\partial f_{n-1}}
					RandomVariableInterface childPartialDerivativeWRTParent = null;
					
					// for expectation operators rely on 
					if(childTreeNode.derivatives.get(i) instanceof ExpectationInformation) {
						ExpectationInformation expectationInformation = (ExpectationInformation) childTreeNode.derivatives.get(0);
						ExpectationType expectationOperator = expectationInformation.expectationType;
						switch (expectationOperator) {
						case UNCONDITIONAL:
							// Implementation of AVERAGE (see https://ssrn.com/abstract=2995695 for details).
							originPartialDerivtivWRTchild = originPartialDerivtivWRTchild.average();
							break;
						case CONDTIONAL:
							// Implementation of CONDITIONAL_EXPECTATION (see https://ssrn.com/abstract=2995695 for details).
							ConditionalExpectationEstimatorInterface estimator = expectationInformation.estimator;
							originPartialDerivtivWRTchild = estimator.getConditionalExpectation(originPartialDerivtivWRTchild);
							break;
						}	
						
						// for expectations the derivative is one
						childPartialDerivativeWRTParent = one;
						
					} else if(childTreeNode.derivatives.get(i) instanceof RandomVariableInterface) {
						
						childPartialDerivativeWRTParent = (RandomVariableInterface) childTreeNode.derivatives.get(i) ;
					
					} else {
						
						// for normal functions just use the partial derivative functions that are given
						childPartialDerivativeWRTParent = childTreeNode.getPartialDerivativeFor(parentTreeNode);
					}
					
					// chain rule - get already existing part of the sum, if it does not exist yet start the sum with zero
					RandomVariableInterface existingChainRuleSum = gradient.getOrDefault(parentID, zero);

					// add existing and new part of the derivative sum 
					RandomVariableInterface chainRuleSum = existingChainRuleSum.addProduct(originPartialDerivtivWRTchild, childPartialDerivativeWRTParent);

					// put result back in gradient
					gradient.put(parentID, chainRuleSum);

					// add parent to ToDo-list to propagate derivatives further downwards
					treeNodesToPropagte.put(parentID, parentTreeNode);
				}

				// if not defined otherwise delete child after derivative has been propagated downwards to parents
				// if no parents existed leave if it for the results
				if(!childTreeNode.parentTreeNodes.isEmpty() && !factory.retainAllTreeNodes) gradient.remove(childID); 
			}

			if(!this.childTreeNodes.isEmpty() && !factory.retainAllTreeNodes) gradient.remove(id);

			return gradient;
		}

		/**
		 * method implements tangent mode of algorithmic differentiation
		 * 
		 * @return {@link Map} with key id and value  dV<sub>id</sub>/dx<sub>this</sub>
		 * */
		private Map<Long, RandomVariableInterface> getAllPartialDerivatives(){

			RandomVariableInterface zero = factory.createRandomVariableNonDifferentiable(-Double.MAX_VALUE, 0.0);
			RandomVariableInterface one = factory.createRandomVariableNonDifferentiable(-Double.MAX_VALUE, 1.0);
						
			Map<Long, RandomVariableInterface> partialDerivatives = new HashMap<>();

			// every child in the operator tree is of the same instance of its parents
			TreeMap<Long, OperatorTreeNode> treeNodesToPropagte = new TreeMap<>();

			// partial derivative with respect to itself
			partialDerivatives.put(id, one);

			// add id of this variable to propagate upwards
			treeNodesToPropagte.put(id, this);

			while(!treeNodesToPropagte.isEmpty()){

				// get and remove smallest ID from treeNodesToPropagate
				Entry<Long, OperatorTreeNode> lowestEntry = treeNodesToPropagte.pollFirstEntry();

				Long parentID = lowestEntry.getKey();
				OperatorTreeNode parentOperatorTreeNode = lowestEntry.getValue();

				final List<OperatorTreeNode> childTreeNodes = parentOperatorTreeNode.childTreeNodes;
				for(int i = 0; i<childTreeNodes.size(); i++) {
					// current child tree node (alsways exists)
					OperatorTreeNode childTreeNode = childTreeNodes.get(i);

					// get current child id
					Long childID = childTreeNode.id;

					// \frac{\partial f}{\partial x} | has to exist by construction!
					RandomVariableInterface parentPartialDerivtivWRTLeaf = partialDerivatives.get(parentID);
					
					// \frac{\partial F}{\partial f}
					RandomVariableInterface childPartialDerivativeWRTParent = null;

					//TODO: Conditional and Unconditional Expectation!
					if(childTreeNode.derivatives.get(0) instanceof ExpectationInformation) {
						ExpectationInformation expectationInformation = (ExpectationInformation) childTreeNode.derivatives.get(0);
						ExpectationType expectationOperator = expectationInformation.expectationType;
						switch (expectationOperator) {
						case UNCONDITIONAL:
							// Implementation of AVERAGE (see https://ssrn.com/abstract=2995695 for details).
							parentPartialDerivtivWRTLeaf = parentPartialDerivtivWRTLeaf.average();
							break;
						case CONDTIONAL:
							// Implementation of CONDITIONAL_EXPECTATION (see https://ssrn.com/abstract=2995695 for details).
							ConditionalExpectationEstimatorInterface estimator = expectationInformation.estimator;
							parentPartialDerivtivWRTLeaf = estimator.getConditionalExpectation(parentPartialDerivtivWRTLeaf);
							break;
						}	
						
						// for expectations the derivative is one
						childPartialDerivativeWRTParent = one;
						
					} else {
						
						// for normal functions just use the partial derivative functions that are given
						childPartialDerivativeWRTParent = childTreeNode.getPartialDerivativeFor(parentOperatorTreeNode);
					}
					// chain rule - get already existing part of the sum
					RandomVariableInterface existingChainRuleSum = partialDerivatives.getOrDefault(childID, zero);

					// add existing and new part of the derivative sum 
					RandomVariableInterface chainRuleSum = existingChainRuleSum.addProduct(childPartialDerivativeWRTParent, parentPartialDerivtivWRTLeaf);

					// put result back in reverseGradient
					partialDerivatives.put(childID, chainRuleSum);

					// add child to ToDo-list to propagate derivatives further upwards
					treeNodesToPropagte.put(childID, childTreeNode);
					}

				// if not defined otherwise delete parent after derivative has been propagated upwards to children
				// if no children existed leave if it for the results
				if(!parentOperatorTreeNode.childTreeNodes.isEmpty() && !factory.retainAllTreeNodes) partialDerivatives.remove(parentID); 
			}
			return partialDerivatives;
		}

		private static OperatorTreeNode treeNodeOf(RandomVariableInterface randomVariable) {
			return randomVariable instanceof RandomVariableDifferentiable ? ((RandomVariableDifferentiable)randomVariable).opteratorTreeNode : null;
		}
	}

	/**
	 * Implementation of <code>RandomVariableDifferentiableInterface</code> using
	 * the forward and reverse algorithmic differentiation (aka. AD and AAD).
	 * 
	 * In order to use the AD functionalities specify this in the factory properties.
	 * 
	 * @author Stefan Sedlmair
	 * @version 1.0
	 * */
	public static class RandomVariableDifferentiable implements RandomVariableDifferentiableInterface {

		private static final long serialVersionUID = 2036109523330671173L;

		private RandomVariableInterface values;
		private final OperatorTreeNode opteratorTreeNode;
		private final RandomVariableDifferentiableFactory factory;

		
		/**
		 * private constructor 
		 * */
		private RandomVariableDifferentiable(RandomVariableInterface randomvariable, List<Object[]> parentInformation, RandomVariableDifferentiableFactory factory) {
			// catch random variable that are of size one and not deterministic!
			if(!randomvariable.isDeterministic() && randomvariable.size() == 1)
				randomvariable = factory.createRandomVariableNonDifferentiable(randomvariable.getFiltrationTime(), randomvariable.get(0));
			
			this.values = valuesOf(randomvariable);
			this.opteratorTreeNode = new OperatorTreeNode(parentInformation, factory);
			this.factory = factory;
		}
		
		/**
		 * Apply any function taking one argument, in order to use it for algorithmic differentiation.
		 * 
		 * @param function {@link DoubleUnaryOperator} f(x)
		 * @param derivativeWrtThis {@link DoubleUnaryOperator} df(x)/dx
		 * @param keepValuesOfThis boolean f'(x) depends on x
		 * @return new {@link RandomVariableDifferentiable}
		 */
		public RandomVariableInterface apply(DoubleUnaryOperator function, DoubleUnaryOperator derivativeWrtThis, boolean keepValuesOfThis) {
			// calculate result
			RandomVariableInterface result = values.apply(function);

			// manage parent information
			ArrayList<Object[]> parentInformation = new ArrayList<>();
			parentInformation.add(new Object[]{this, derivativeWrtThis, keepValuesOfThis});

			// return new random variable
			return new RandomVariableDifferentiable(result, parentInformation, getFactory());
		}
		
		/**
		 * Apply any function taking two arguments, in order to use it for algorithmic differentiation.
		 * 
		 * @param function {@link DoubleBinaryOperator} f(x, y)
		 * @param argument1 {@link RandomVariableInterface} y
		 * @param derivativeWrtThis {@link DoubleBinaryOperator} df(x,y)/dx
		 * @param derivativeWrtArgument1 {@link DoubleBinaryOperator} df(x,y)/dy
		 * @param keepValuesOfThis boolean f'(x,y) depends on x
		 * @param keepValuesOfArgument1 boolean f'(x,y) depends on y
		 * @return new {@link RandomVariableDifferentiable}
		 */
		public RandomVariableInterface apply(DoubleBinaryOperator function, RandomVariableInterface argument1,
				DoubleBinaryOperator derivativeWrtThis, DoubleBinaryOperator derivativeWrtArgument1,
				boolean keepValuesOfThis ,boolean keepValuesOfArgument1) {

			// calculate result
			RandomVariableInterface result = values.apply(function, argument1);

			// manage parent information
			ArrayList<Object[]> parentInformation = new ArrayList<>();
			parentInformation.add(new Object[]{this, derivativeWrtThis, keepValuesOfThis});
			parentInformation.add(new Object[]{argument1, derivativeWrtArgument1, keepValuesOfArgument1});

			// return new random variable
			return new RandomVariableDifferentiable(result, parentInformation, getFactory());
		}

		/**
		 * Apply any function taking three arguments, in order to use it for algorithmic differentiation.
		 * 
		 * @param function {@link DoubleTernaryOperator} f(x, y, z)
		 * @param argument1 {@link RandomVariableInterface} y
		 * @param argument2 {@link RandomVariableInterface} z
		 * @param derivativeWrtThis {@link DoubleTernaryOperator} df(x, y, z)/dx
		 * @param derivativeWrtArgument1 {@link DoubleTernaryOperator} df(x, y, z)/dy
		 * @param derivativeWrtArgument2 {@link DoubleTernaryOperator} df(x, y, z)/dz
		 * @param keepValuesOfThis boolean f'(x,y,z) depends on x
		 * @param keepValuesOfArgument1 boolean f'(x,y,z) depends on y 
		 * @param keepValuesOfArgument2 boolean f'(x,y,z) depends on z
		 * @return new {@link RandomVariableDifferentiable}
		 * */
		public RandomVariableInterface apply(DoubleTernaryOperator function, 
				RandomVariableInterface argument1, RandomVariableInterface argument2,
				DoubleTernaryOperator derivativeWrtThis, DoubleTernaryOperator derivativeWrtArgument1, DoubleTernaryOperator derivativeWrtArgument2,
				boolean keepValuesOfThis, boolean keepValuesOfArgument1, boolean keepValuesOfArgument2) {
			// calculate result
			RandomVariableInterface result = values.apply(function, argument1, argument2);

			// manage parent information
			ArrayList<Object[]> parentInformation = new ArrayList<>();
			parentInformation.add(new Object[]{this, derivativeWrtThis, keepValuesOfThis});
			parentInformation.add(new Object[]{argument1, derivativeWrtArgument1, keepValuesOfArgument1});
			parentInformation.add(new Object[]{argument2, derivativeWrtArgument2, keepValuesOfArgument2});

			// return new random variable
			return new RandomVariableDifferentiable(result, parentInformation, getFactory());
		}

		/**
		 * Extracts the values argument if parameter is of instance {@link RandomVariableDifferentiable}
		 * 
		 * @param randomVariable
		 * @return values of randomVariable if instance of {@link RandomVariableDifferentiable}
		 * */
		private static RandomVariableInterface valuesOf(RandomVariableInterface randomVariable) {
			return randomVariable instanceof RandomVariableDifferentiable ? ((RandomVariableDifferentiable)randomVariable).values : randomVariable;
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#equals(net.finmath.stochastic.RandomVariableInterface)
		 */
		@Override
		public boolean equals(RandomVariableInterface randomVariable) {
			return getValues().equals(randomVariable);
		}

		private RandomVariableInterface getValues() {
			return values;
		}

		private RandomVariableDifferentiableFactory getFactory() {
			return factory;
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

		@Override
		public Double doubleValue() {
			return getValues().doubleValue();
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
			return ((RandomVariableDifferentiable) getValues()).getValues().getQuantile(quantile, probabilities);
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getQuantileExpectation(double, double)
		 */
		@Override
		public double getQuantileExpectation(double quantileStart, double quantileEnd) {
			return ((RandomVariableDifferentiable) getValues()).getValues().getQuantileExpectation(quantileStart, quantileEnd);
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

		@Override
		public RandomVariableInterface cache() {
			values = getValues().cache();
			return this;
		}

		/*
		 * The following methods are operations with are differentiable.
		 */
		
		@Override
		public RandomVariableInterface apply(DoubleUnaryOperator operator) {
			// get finite difference step size
			double finiteDifferencesStepSize = getFactory().finiteDifferencesStepSize;
			double epsilonX = (this.getStandardDeviation() > 0.0 ? this.getStandardDeviation() : 1.0) * finiteDifferencesStepSize;

			// apply central finite differences on unknown operator
			return apply(operator,
						 x -> (operator.applyAsDouble(x+epsilonX) - operator.applyAsDouble(x-epsilonX))/(2.0*epsilonX),
						 true);
		}

		@Override
		public RandomVariableInterface apply(DoubleBinaryOperator operator, RandomVariableInterface argument) {
			// get finite difference step size
			double finiteDifferencesStepSize = getFactory().finiteDifferencesStepSize;
			double epsilonX = (this.getStandardDeviation() > 0.0 ? this.getStandardDeviation() : 1.0) * finiteDifferencesStepSize;
			double epsilonY = (argument.getStandardDeviation() > 0.0 ? argument.getStandardDeviation() : 1.0) * finiteDifferencesStepSize;

			// apply central finite differences on unknown operator
			return apply(operator, argument,
						 (x,y) -> (operator.applyAsDouble(x+epsilonX,y) - operator.applyAsDouble(x-epsilonX,y))/(2.0*epsilonX),
						 (x,y) -> (operator.applyAsDouble(x,y+epsilonY) - operator.applyAsDouble(x,y-epsilonY))/(2.0*epsilonY),
						 true, true);
		}

		@Override
		public RandomVariableInterface apply(DoubleTernaryOperator operator, RandomVariableInterface argument1,
				RandomVariableInterface argument2) {
			// get finite difference step size
			double finiteDifferencesStepSize = getFactory().finiteDifferencesStepSize;
			double epsilonX = (this.getStandardDeviation() > 0.0 ? this.getStandardDeviation() : 1.0) * finiteDifferencesStepSize;
			double epsilonY = (argument1.getStandardDeviation() > 0.0 ? argument1.getStandardDeviation() : 1.0) * finiteDifferencesStepSize;
			double epsilonZ = (argument2.getStandardDeviation() > 0.0 ? argument2.getStandardDeviation() : 1.0) * finiteDifferencesStepSize;
			
			// apply central finite differences on unknown operator
			return apply(operator, argument1, argument2,
						 (x,y,z) -> (operator.applyAsDouble(x+epsilonX,y,z) - operator.applyAsDouble(x-epsilonX,y,z))/(2.0*epsilonX),
						 (x,y,z) -> (operator.applyAsDouble(x,y+epsilonY,z) - operator.applyAsDouble(x,y-epsilonY,z))/(2.0*epsilonY),
						 (x,y,z) -> (operator.applyAsDouble(x,y,z+epsilonZ) - operator.applyAsDouble(x,y,z-epsilonZ))/(2.0*epsilonZ),
						 true, true, true);
		}

		@Override
		public RandomVariableInterface floor(double floor) {
			return apply(x -> FastMath.max(x, floor), 
						 x -> x < floor ? 0.0 : 1.0, 
						 true);
		}
		
		@Override
		public RandomVariableInterface cap(double cap) {
			return apply(x -> FastMath.min(x, cap), 
					 	 x -> x < cap ? 1.0 : 0.0, 
					     true);
		}

		@Override
		public RandomVariableInterface add(double value) {
			return apply(x -> x + value, 
						 x -> 1.0, 
						 false);
		}

		@Override
		public RandomVariableInterface sub(double value) {
			return add(-value);
		}

		@Override
		public RandomVariableInterface mult(double value) {
			return apply(x -> x * value, 
						 x -> value, 
						 false);
		}

		@Override
		public RandomVariableInterface div(double value) {
			return mult(1.0/value);
		}

		@Override
		public RandomVariableInterface pow(double exponent) {
			return apply(x -> FastMath.pow(x, exponent), 
						 x -> exponent * FastMath.pow(x, exponent - 1.0), 
						 true);
		}

		@Override
		public RandomVariableInterface average() {
			
			List<Object[]> parentInformation = new ArrayList<>();			
			parentInformation.add(new Object[]{this, new ExpectationInformation(ExpectationType.UNCONDITIONAL) , false, true});

			return new RandomVariableDifferentiable(values.average(), parentInformation, factory);
		}

		@Override
		public RandomVariableInterface getConditionalExpectation(
				ConditionalExpectationEstimatorInterface conditionalExpectationOperator) {			
			List<Object[]> parentInformation = new ArrayList<>();			
			parentInformation.add(new Object[]{this, new ExpectationInformation(ExpectationType.UNCONDITIONAL, conditionalExpectationOperator) , false, true});

			return new RandomVariableDifferentiable(values.average(), parentInformation, factory);
		}
		
		@Override
		public RandomVariableInterface squared() {
			return apply(x -> x * x, 
						 x -> 2.0 * x, 
						 true);
		}

		@Override
		public RandomVariableInterface sqrt() {
			return apply(FastMath::sqrt, 
						 x -> 0.5 / FastMath.sqrt(x), 
						 true);
		}

		@Override
		public RandomVariableInterface exp() {
			return apply(FastMath::exp, 
						 FastMath::exp, 
						 true);
		}

		@Override
		public RandomVariableInterface log() {
			return apply(FastMath::log, 
						 x -> 1.0/x, 
						 true);
		}

		@Override
		public RandomVariableInterface sin() {
			return apply(FastMath::sin, 
						 FastMath::cos, 
						 true);
		}

		@Override
		public RandomVariableInterface cos() {
			return apply(FastMath::cos, 
						 x -> -FastMath.sin(x), 
						 true);
		}

		@Override
		public RandomVariableInterface add(RandomVariableInterface randomVariable) {
			return apply((x,y) -> x + y, randomVariable, 
						 (x,y) -> 1.0, 
						 (x,y) -> 1.0, 
						 false, false);
		}

		@Override
		public RandomVariableInterface sub(RandomVariableInterface randomVariable) {
			return apply((x,y) -> x - y, randomVariable, 
						 (x,y) -> +1.0, 
						 (x,y) -> -1.0, 
						 false, false);
		}

		@Override
		public RandomVariableInterface mult(RandomVariableInterface randomVariable) {
			return apply((x,y) -> x * y, randomVariable, 
						 (x,y) -> y, 
						 (x,y) -> x, 
						 true, true);
		}

		@Override
		public RandomVariableInterface div(RandomVariableInterface randomVariable) {
			return apply((x,y) -> x / y, randomVariable, 
						 (x,y) -> 1.0 / y, 
						 (x,y) -> -x / (y*y),
						 true, true);
		}

		@Override
		public RandomVariableInterface cap(RandomVariableInterface cap) {
			return apply(FastMath::min, cap, 
						 (x,y) -> x < y ? 1.0 : 0.0, 
						 (x,y) -> x < y ? 0.0 : 1.0, 
						 true, true);
		}

		@Override
		public RandomVariableInterface floor(RandomVariableInterface floor) {
			return apply(FastMath::max, floor, 
						 (x,y) -> x > y ? 1.0 : 0.0, 
						 (x,y) -> x > y ? 0.0 : 1.0, 
						 true, true);
		}

		@Override
		public RandomVariableInterface accrue(RandomVariableInterface rate, double periodLength) {
			return apply((x,y) -> x * (1.0 + y * periodLength), rate,
 						 (x,y) -> 1.0 + y * periodLength, 
 						 (x,y) -> x * periodLength, 
 						 true, true);
		}

		@Override
		public RandomVariableInterface discount(RandomVariableInterface rate, double periodLength) {
			return apply((x,y) -> x / (1.0 + y * periodLength), rate,
						 (x,y) -> 1.0/ (1.0 + y * periodLength), 
						 (x,y) -> -1.0 * x * periodLength / FastMath.pow(1.0 + y * periodLength, 2), 
						 true, true);
		}

		@Override
		public RandomVariableInterface barrier(RandomVariableInterface trigger,
				RandomVariableInterface valueIfTriggerNonNegative, RandomVariableInterface valueIfTriggerNegative) {
			double epsilon = factory.barrierDiracWidth * trigger.getStandardDeviation();
			
			if(epsilon > 0.0) 
				return apply((x,y,z) -> x >= 0.0 ? y : z, valueIfTriggerNonNegative, valueIfTriggerNegative,
						(x,y,z) -> (y - z)*((x + epsilon/2) >= 0 ? 1.0 : 0.0)*((x - epsilon/2) >= 0 ? 0.0 : 1.0)/epsilon,
						(x,y,z) -> x >= 0.0 ? 1.0 : 0.0,
						(x,y,z) -> x >= 0.0 ? 0.0 : 1.0,
						true, true, true);
			
			if(epsilon == 0.0)
				return apply((x,y,z) -> x >= 0.0 ? y : z, valueIfTriggerNonNegative, valueIfTriggerNegative,
						(x,y,z) -> x == 0 ? (z < y ? Double.POSITIVE_INFINITY : ( z == y ? 0.0 : Double.NEGATIVE_INFINITY)) : 0.0 ,
						(x,y,z) -> x >= 0.0 ? 1.0 : 0.0,
						(x,y,z) -> x >= 0.0 ? 0.0 : 1.0,
						true, true, true);
			else throw new IllegalArgumentException("Epsilon shoud never be negative!");
		}

		@Override
		public RandomVariableInterface barrier(RandomVariableInterface trigger,
				RandomVariableInterface valueIfTriggerNonNegative, double valueIfTriggerNegative) {
			double epsilon = factory.barrierDiracWidth * trigger.getStandardDeviation();
			
			if(epsilon > 0.0) 
				return apply((x,y) -> x >= 0.0 ? y : valueIfTriggerNegative, valueIfTriggerNonNegative,
							 (x,y) -> (y - valueIfTriggerNegative)*((x + epsilon/2) >= 0 ? 1.0 : 0.0)*((x - epsilon/2) >= 0 ? 0.0 : 1.0)/epsilon,
							 (x,y) -> x >= 0.0 ? 1.0 : 0.0,
							 true, true);
			
			if(epsilon == 0.0)
				return apply((x,y) -> x >= 0.0 ? y : valueIfTriggerNegative, valueIfTriggerNonNegative,
						 	 (x,y) -> x == 0 ? (valueIfTriggerNegative < y ? Double.POSITIVE_INFINITY : ( valueIfTriggerNegative == y ? 0.0 : Double.NEGATIVE_INFINITY)) : 0.0 ,
						 	 (x,y) -> x >= 0.0 ? 1.0 : 0.0,
						 	 true, true);
			else throw new IllegalArgumentException("Epsilon shoud never be negative!");
		}

		@Override
		public RandomVariableInterface invert() {
			return pow(-1.0);
		}

		@Override
		public RandomVariableInterface abs() {
			return apply(FastMath::abs,
						 x -> x < 0.0 ? -1.0 : x > 0 ? 1.0 : 0.0, 
						 true);
		}

		@Override
		public RandomVariableInterface addProduct(RandomVariableInterface factor1, double factor2) {
			return apply((x,y) -> x + y * factor2, factor1,
						 (x,y) -> 1.0, 
						 (x,y) -> factor2, 
						 false, false);
		}

		@Override
		public RandomVariableInterface addProduct(RandomVariableInterface factor1, RandomVariableInterface factor2) {
			return apply((x,y,z) -> x + y * z, factor1, factor2,
						 (x,y,z) -> 1.0,
						 (x,y,z) -> z, 
						 (x,y,z) -> y, 
						 false, true, true);
		}

		@Override
		public RandomVariableInterface addRatio(RandomVariableInterface numerator,
				RandomVariableInterface denominator) {
			return apply((x,y,z) -> x + y / z, numerator, denominator,
						 (x,y,z) -> 1.0,
						 (x,y,z) -> +1.0/z, 
						 (x,y,z) -> -y/(z*z), 
						 false, true, true);
		}

		@Override
		public RandomVariableInterface subRatio(RandomVariableInterface numerator,
				RandomVariableInterface denominator) {
			return apply((x,y,z) -> x - y / z, numerator, denominator,
						 (x,y,z) -> 1.0,
						 (x,y,z) -> -1.0/z, 
						 (x,y,z) -> +y/(z*z), 
						 false, true, true);
		}
		
		@Override
		public RandomVariableInterface isNaN() {
			return getValues().isNaN();
		}

		@Override
		public Map<Long, RandomVariableInterface> getGradientOf(Set<Long> targetIDs) {
			if(targetIDs != null) throw new UnsupportedOperationException();
			return opteratorTreeNode.getGradient();
		}

		public Map<Long, RandomVariableInterface> getPartialDerivativesOf(Set<Long> targetIDs) {
			if(targetIDs != null) throw new UnsupportedOperationException();
			if(!factory.enableAD) throw new UnsupportedOperationException();
			return opteratorTreeNode.getAllPartialDerivatives();
		}

		@Override
		public Long getID() {
			return opteratorTreeNode.id;
		}

		@Override
		public IntToDoubleFunction getOperator() {
			return values.getOperator();
		}

		@Override
		public DoubleStream getRealizationsStream() {
			return Arrays.stream(getRealizations());
		}
		
		
		/**
		 * Method 
		 * */
		private static RandomVariableInterface catchWronglyNonDeterministicRandomVariable(RandomVariableInterface randomVariable) {
			if(!randomVariable.isDeterministic() && VectorAlgbra.isAllEntriesEqual(randomVariable.getRealizations())){
				double time = randomVariable.getFiltrationTime();
				double value = randomVariable.get(0);

				if(randomVariable instanceof RandomVariableDifferentiable) {
					RandomVariableDifferentiable rvAutoDiff = ((RandomVariableDifferentiable) randomVariable); 
					rvAutoDiff.values = rvAutoDiff.factory.createRandomVariableNonDifferentiable(time, value);
				} else {
					randomVariable = new RandomVariable(time, value);
				}
			}
			return randomVariable;
		}

		
		private static RandomVariableInterface nullToNaN(RandomVariableInterface X) {
			RandomVariableInterface nan = new RandomVariable(Double.NaN);
			return X == null ? nan : X.cache();
		}

		/**
		 * executes {@link DoubleUnaryOperator} and ignores null values
		 * @return function value, NaN if null in dependent variable
		 */
		public static RandomVariableInterface apply(DoubleUnaryOperator function, List<RandomVariableInterface> X) {
			RandomVariableInterface result = nullToNaN(X.get(0)).apply(function);
			 //catch non-deterministic random variables with length 1
			result = catchWronglyNonDeterministicRandomVariable(result);
			
			return result;
		}
			
		/**
		 * executes {@link DoubleBinaryOperator} and ignores null values
		 * @return function value, NaN if null in dependent variable
		 */
		public static RandomVariableInterface apply(DoubleBinaryOperator function, List<RandomVariableInterface> X) {
			RandomVariableInterface result = nullToNaN(X.get(0)).apply(function, nullToNaN(X.get(1)));
			 //catch non-deterministic random variables with length 1
			result = catchWronglyNonDeterministicRandomVariable(result);
			
			return result;
		}

		/**
		 * executes {@link DoubleTernaryOperator} and ignores null values
		 * @return function value, NaN if null in dependent variable
		 */
		public static RandomVariableInterface apply(DoubleTernaryOperator function, List<RandomVariableInterface> X) {
			RandomVariableInterface result = nullToNaN(X.get(0)).apply(function, nullToNaN(X.get(1)), nullToNaN(X.get(2)));
			 //catch non-deterministic random variables with length 1
			result = catchWronglyNonDeterministicRandomVariable(result);
			
			return result;
		}
		
		@Override
		public String toString() {
			return "RandomVariableDifferentiable [values=" + values + ", ID=" + getID() + "]";
		}
	}
	
	private static enum ExpectationType{
		UNCONDITIONAL, CONDTIONAL
	}
	
	private static class ExpectationInformation{
		public final ExpectationType expectationType;
		public final ConditionalExpectationEstimatorInterface estimator;
		
		public ExpectationInformation(ExpectationType expectationType, ConditionalExpectationEstimatorInterface estimator) {
			this.expectationType = expectationType;
			this.estimator = estimator;
		}
		
		public ExpectationInformation(ExpectationType expectationType) {
			this(expectationType, null);
		}
	}
}
