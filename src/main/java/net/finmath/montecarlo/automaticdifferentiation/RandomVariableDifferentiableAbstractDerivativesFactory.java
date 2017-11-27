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
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.apache.commons.math3.util.FastMath;

import net.finmath.functions.DoubleTernaryOperator;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.RandomVariable;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.stochastic.ConditionalExpectationEstimatorInterface;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * 
 * 
 * @author Stefan Sedlmair
 * @version 1.0
 */
public class RandomVariableDifferentiableAbstractDerivativesFactory extends AbstractRandomVariableDifferentiableFactory {

	private final static AtomicLong nextIdentifier = new AtomicLong(0); 

	private final double finiteDifferencesStepSize;

	private final boolean enableAD;
	private final boolean enableAAD;
	private final boolean retainAllTreeNodes;


	/**
	 * @param randomVariableFactoryForNonDifferentiable
	 */
	public RandomVariableDifferentiableAbstractDerivativesFactory(AbstractRandomVariableFactory randomVariableFactoryForNonDifferentiable, Map<String, Object> properties) {
		super(randomVariableFactoryForNonDifferentiable);

		// step-size for the usage of finite differences if no analytic derivative is given
		this.finiteDifferencesStepSize 	= (double) properties.getOrDefault("finiteDifferencesStepSize", 1E-8);

		// enable AD by keeping track of upwards 
		this.enableAD 				= (boolean) properties.getOrDefault("enableAD", true);
		this.enableAAD 				= (boolean) properties.getOrDefault("enableAAD", true);
		this.retainAllTreeNodes 	= (boolean) properties.getOrDefault("retainAllTreeNodes", false);
	}

	public RandomVariableDifferentiableAbstractDerivativesFactory(AbstractRandomVariableFactory randomVariableFactoryForNonDifferentiable) {
		this(randomVariableFactoryForNonDifferentiable, new HashMap<>());
	}

	public RandomVariableDifferentiableAbstractDerivativesFactory() {
		this(new RandomVariableFactory(), new HashMap<>());
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableAbstractFactory#createRandomVariable(double, double)
	 */
	@Override
	public RandomVariableDifferentiableInterface createRandomVariable(double time, double value) {
		RandomVariableInterface randomvariable = super.createRandomVariableNonDifferentiable(time, value);
		return new RandomVariableDifferentiableAbstract(randomvariable, null, null, this);
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableAbstractFactory#createRandomVariable(double, double[])
	 */
	@Override
	public RandomVariableDifferentiableInterface createRandomVariable(double time, double[] values) {		
		RandomVariableInterface randomvariable = super.createRandomVariableNonDifferentiable(time, values);
		return new RandomVariableDifferentiableAbstract(randomvariable, null, null, this);
	}

	@Override
	public RandomVariableInterface createRandomVariableNonDifferentiable(double time, double[] values) {
		// catch any randomVariables beeing generated that are pseudo-stochastic
		if(values.length == 1) return super.createRandomVariableNonDifferentiable(time, values[0]); 

		// else use standard implementation
		return super.createRandomVariableNonDifferentiable(time, values);
	}


	/**
	 * Novel version of Operator Tree Nodes
	 * Manages the dependencies of in the operator tree and the calculation of the derivatives
	 * 
	 * Can hold either AD
	 * 
	 * @author Stefan Sedlmair
	 * @author Christian Fries
	 * @version 2.0
	 * */
	private static class OperatorTreeNode{
		private final long id;
		private final RandomVariableDifferentiableAbstractDerivativesFactory factory;

		private final List<OperatorTreeNode> parentTreeNodes;
		private final List<OperatorTreeNode> childTreeNodes;

		private final PartialDerivativeFunction derivatives;	

		/**
		 * Note derivativeWrtParentRandomVariable can hold {@link ExpectationInformation}.
		 * 
		 * @param parentInfromation list of object arrays. Has to be of the following order: <lu><li>ParentRandomVariable</li><li>derivativeWrtParent</li><li>keepValuesOfParent</li></lu>  
		 * @param factory {@link AbstractRandomVariableDifferentiableAbstractFactory} to generate random variables
		 * */
		public OperatorTreeNode(List<OperatorTreeNode> parents, PartialDerivativeFunction partialDerivativeFunction, RandomVariableDifferentiableAbstractDerivativesFactory factory) {
			// get identifier
			this.id = nextIdentifier.getAndIncrement();
			
			// get factory
			this.factory = factory;

			// initialize parent tree nodes if needed
			if(factory.enableAAD) 	this.parentTreeNodes = parents;
			else 					this.parentTreeNodes = null;

			// only connect to children if AD is enabled
			if(factory.enableAD) {
				// start with an empty list
				this.childTreeNodes = Collections.synchronizedList(new ArrayList<OperatorTreeNode>());
				// fill this into lists of parents 
				if(parents != null)
					for(OperatorTreeNode parentTreeNode : parents) 
						if(parentTreeNode != null) 
							parentTreeNode.childTreeNodes.add(this);
			}
			else this.childTreeNodes = null;
			
			// in any case initialize partial derivative function (holds values of parents)
			if(factory.enableAD || factory.enableAAD) this.derivatives = partialDerivativeFunction;
			// in case neither algorithmic differentiation technique is selected do not save partial derivative function either
			else									  this.derivatives = null;

		}

		/**
		 * method implements reverse mode of algorithmic differentiation
		 * 
		 * @return {@link Map} with key id and value  dV<sub>this</sub>/dx<sub>id</sub>
		 * */
		private Map<Long, RandomVariableInterface> getGradient() {
			
			Map<Long, RandomVariableInterface> gradient = Collections.synchronizedMap(new HashMap<>());

			// every child in the operator tree is of the same instance of its parents
//			TreeMap<Long, OperatorTreeNode> treeNodesToPropagte = new TreeMap<>();
			// thread save treeMap
			ConcurrentSkipListMap<Long, OperatorTreeNode> treeNodesToPropagte = new ConcurrentSkipListMap<>();

			// partial derivative with respect to itself
			gradient.put(id, PartialDerivativeFunction.one);

			// add id of this variable to propagate downwards
			treeNodesToPropagte.put(id, this);			

			// sequentially go through all treeNodes in the respective operator tree and propergate the derivatives downwards
			while(!treeNodesToPropagte.isEmpty()){

				// get and remove highest ID from treeNodesToPropagate
				Entry<Long, OperatorTreeNode> highestEntry = treeNodesToPropagte.pollLastEntry();

				Long childID = highestEntry.getKey();
				OperatorTreeNode childTreeNode = highestEntry.getValue();
				
				final List<OperatorTreeNode> parentTreeNodes = childTreeNode.parentTreeNodes;
								
				if(parentTreeNodes == null) continue;
				
				propergateTreeNodesThroughOperatorTree(gradient, childTreeNode, parentTreeNodes, treeNodesToPropagte);
				
				// if not defined otherwise delete child after derivative has been propagated downwards to parents
				if(!factory.retainAllTreeNodes) gradient.remove(childID); 
			}
			return gradient;
		}

		/**
		 * Propagate the derivatives along any direction of the Operator tree:
		 * 
		 * <center> works for AD and AAD </center>
		 * 
		 * @param derivatives {@link Map} of all derivatives already calculated
		 * @param treeNodeOfOrigin the {@link OperatorTreeNode} from which we are propagating the derivative from
		 * @param treeNodes the {@link OperatorTreeNode}s to which the derivatives will be propagated
		 * @param treeNodesToPropagte {@link ConcurrentSkipListMap} of {@link OperatorTreeNode}s that still have to be propagated 
		 * */
		private static void propergateTreeNodesThroughOperatorTree(Map<Long, RandomVariableInterface> derivatives, 
				OperatorTreeNode treeNodeOfOrigin, List<OperatorTreeNode> treeNodes, ConcurrentSkipListMap<Long, OperatorTreeNode> treeNodesToPropagte) {
						
			final long originID = treeNodeOfOrigin.id;
			
			// derivative from root/leaf of tree wrt to treeNodeOfOrigin
			final RandomVariableInterface derivative = derivatives.get(originID);
			
			// each treeNode can be propagated independently 
			IntStream.range(0, treeNodes.size()).forEach(treeNodeIndex -> {
					OperatorTreeNode treeNode = treeNodes.get(treeNodeIndex);
			
					// if parentTreeNode is null, derivative is zero, thus continue with next treeNode
					if(treeNode == null) return;

					// get the current parent id
					final Long ID = treeNode.id;
					
					// get partial derivative of treeNodeOfOrigin wrt the treeNode and multiply it with its derivative
					RandomVariableInterface newAddendOfChainRuleSum = (originID > ID) ? 
							/*AAD*/	treeNodeOfOrigin.getDerivativeProduct(treeNode, derivative):
							/* AD*/	treeNode.getDerivativeProduct(treeNodeOfOrigin, derivative);
							
					
					// chain rule - get already existing part of the sum, if it does not exist yet start the sum with zero
					RandomVariableInterface existingChainRuleSum = derivatives.getOrDefault(ID, PartialDerivativeFunction.zero);

					// add existing and new part of the derivative sum 
					RandomVariableInterface chainRuleSum = existingChainRuleSum.add(newAddendOfChainRuleSum);

					// put result back in derivatives
					derivatives.put(ID, chainRuleSum);

					// add parent to ToDo-list to propagate derivatives further downwards
					treeNodesToPropagte.put(ID, treeNode);
				});
		}

		private RandomVariableInterface getDerivativeProduct(OperatorTreeNode treeNode, RandomVariableInterface derivative) {
			// if no derivative function exists the partial derivative w.r.t. some parameter will always be zero
			if(derivatives == null) return PartialDerivativeFunction.zero;
						
			// map TreeNode to index
			int parameterIndex = indexOfTreeNodeInParentTreeNodes(treeNode); 
			
			// if parameterIndex smaller zero, this value is independent of this ID
			if(parameterIndex < 0) return PartialDerivativeFunction.zero;
			
			// calculate the multiplication for the chain rule sum
			return derivatives.getDerivativeProduct(parameterIndex, derivative);
		}
		
		private int indexOfTreeNodeInParentTreeNodes(OperatorTreeNode treeNode){
			for(int index = 0; index < parentTreeNodes.size(); index++)
				if(parentTreeNodes.get(index).id == treeNode.id) return index;
			return -1;
		}

		/**
		 * method implements tangent mode of algorithmic differentiation
		 * 
		 * @return {@link Map} with key id and value  dV<sub>id</sub>/dx<sub>this</sub>
		 * */
		private Map<Long, RandomVariableInterface> getAllPartialDerivatives(){
			
			Map<Long, RandomVariableInterface> partialDerivatives = new HashMap<>();

			// every child in the operator tree is of the same instance of its parents
			ConcurrentSkipListMap<Long, OperatorTreeNode> treeNodesToPropagte = new ConcurrentSkipListMap<>();

			// partial derivative with respect to itself
			partialDerivatives.put(id, PartialDerivativeFunction.one);

			// add id of this variable to propagate upwards
			treeNodesToPropagte.put(id, this);

			while(!treeNodesToPropagte.isEmpty()){

				// get and remove smallest ID from treeNodesToPropagate
				Entry<Long, OperatorTreeNode> lowestEntry = treeNodesToPropagte.pollFirstEntry();

				Long parentID = lowestEntry.getKey();
				OperatorTreeNode parentOperatorTreeNode = lowestEntry.getValue();

				final List<OperatorTreeNode> childTreeNodes = parentOperatorTreeNode.childTreeNodes;
				
				if(childTreeNodes.isEmpty()) continue;
				
				propergateTreeNodesThroughOperatorTree(partialDerivatives, parentOperatorTreeNode, childTreeNodes, treeNodesToPropagte);
	
				// if not defined otherwise delete parent after derivative has been propagated upwards to children
				if(!factory.retainAllTreeNodes) partialDerivatives.remove(parentID); 
			}
			return partialDerivatives;
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
	public static class RandomVariableDifferentiableAbstract implements RandomVariableDifferentiableInterface {

		private static final long serialVersionUID = 2036109523330671173L;

		private 	  RandomVariableInterface values;
		private final OperatorTreeNode opteratorTreeNode;
		private final RandomVariableDifferentiableAbstractDerivativesFactory factory;


		/**
		 * Constructor for generating a {@link RandomVariableDifferentiableAbstract} 
		 * 
		 * @param randomvariable {@link RandomVariableInterface} to store values
		 * @param parents {@link List} of {@link RandomVariableInterface}s that where arguments of the function which resulted in randomVariable (<code>null</code> if non exist)
		 * @param partialDerivativeFunction {@link PartialDerivativeFunction} defining the way the partial derivatives are calculated (if <code>null</code> partial derivative will always be zero)
		 * @param factory {@link RandomVariableDifferentiableAbstractDerivativesFactory} factory to construct new {@link RandomVariableInterface}s
		 * */
		public RandomVariableDifferentiableAbstract(RandomVariableInterface randomvariable, List<RandomVariableInterface> parents, PartialDerivativeFunction partialDerivativeFunction, 
				RandomVariableDifferentiableAbstractDerivativesFactory factory) {

			// extract the tree nodes from 
			List<OperatorTreeNode> parentTreeNodes = null;
			if(parents != null) {
				parentTreeNodes = new ArrayList<>();
				for(int parentIndex = 0; parentIndex < parents.size(); parentIndex++)
					parentTreeNodes.add(parentIndex, treeNodeOf(parents.get(parentIndex)));
				// free memory
				parents = null;
			}
			
			// set values 
			this.values = valuesOf(randomvariable);
			
			//
			this.opteratorTreeNode = new OperatorTreeNode(parentTreeNodes, partialDerivativeFunction, factory);
			this.factory = factory;
		}
		
		/**
		 * Extracts the values argument if parameter is of instance {@link RandomVariableDifferentiableAbstract}
		 * 
		 * @param randomVariable
		 * @return values of randomVariable if instance of {@link RandomVariableDifferentiableAbstract}
		 * */
		private static RandomVariableInterface valuesOf(RandomVariableInterface randomVariable) {
			return randomVariable instanceof RandomVariableDifferentiableAbstract ? ((RandomVariableDifferentiableAbstract)randomVariable).values : randomVariable;
		}

		private static OperatorTreeNode treeNodeOf(RandomVariableInterface randomVariable) {
			return randomVariable instanceof RandomVariableDifferentiableAbstract ? ((RandomVariableDifferentiableAbstract)randomVariable).opteratorTreeNode : null;
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

		private RandomVariableDifferentiableAbstractDerivativesFactory getFactory() {
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
			return ((RandomVariableDifferentiableAbstract) getValues()).getValues().getQuantile(quantile, probabilities);
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getQuantileExpectation(double, double)
		 */
		@Override
		public double getQuantileExpectation(double quantileStart, double quantileEnd) {
			return ((RandomVariableDifferentiableAbstract) getValues()).getValues().getQuantileExpectation(quantileStart, quantileEnd);
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
			return new RandomVariableDifferentiableAbstract(
					values.apply(operator),
					Arrays.asList(this),
					new PartialDerivativeFunction(this) {

						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return X.add(epsilonX).apply(operator).sub(X.sub(epsilonX).apply(operator)).div(2.0 * epsilonX);
							default:
								return null;
							}
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface apply(DoubleBinaryOperator operator, RandomVariableInterface argument) {
			// get finite difference step size
			double finiteDifferencesStepSize = getFactory().finiteDifferencesStepSize;
			double epsilonX = (this.getStandardDeviation() > 0.0 ? this.getStandardDeviation() : 1.0) * finiteDifferencesStepSize;
			double epsilonY = (argument.getStandardDeviation() > 0.0 ? argument.getStandardDeviation() : 1.0) * finiteDifferencesStepSize;

			// apply central finite differences on unknown operator
			return new RandomVariableDifferentiableAbstract(
					values.apply(operator, argument),
					Arrays.asList(this, argument),
					new PartialDerivativeFunction(this, argument) {
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return X.add(epsilonX).apply(operator, Y).sub(X.sub(epsilonX).apply(operator, Y)).div(2.0 * epsilonX);
							case 1:
								return X.apply(operator, Y.add(epsilonY)).sub(X.apply(operator, Y.sub(epsilonY))).div(2.0 * epsilonY);
							default:
								return null;
							}
						}
					}, 
					getFactory());
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
			return new RandomVariableDifferentiableAbstract(
					values.apply(operator, argument1, argument2),
					Arrays.asList(this, argument1, argument2),
					new PartialDerivativeFunction(this, argument1, argument2) {
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return X.add(epsilonX).apply(operator, Y, Z).sub(X.sub(epsilonX).apply(operator, Y, Z)).div(2.0 * epsilonX);
							case 1:
								return X.apply(operator, Y.add(epsilonY), Z).sub(X.apply(operator, Y.sub(epsilonY), Z)).div(2.0 * epsilonY);
							case 2:
								return X.apply(operator, Y, Z.add(epsilonZ)).sub(X.apply(operator, Y, Z.sub(epsilonZ))).div(2.0 * epsilonZ);

							default:
								return null;
							}
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface floor(double floor) {
			return floor(factory.createRandomVariableNonDifferentiable(0.0, floor));
		}

		@Override
		public RandomVariableInterface cap(double cap) {
			return cap(factory.createRandomVariableNonDifferentiable(getFiltrationTime(), cap));
		}

		@Override
		public RandomVariableInterface add(double value) {
			RandomVariableInterface result = values.add(value);

			PartialDerivativeFunction partialDerivativeFunction = new PartialDerivativeFunction(null) {	
				@Override
				public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {

					if(parameterIndex != 0) return null;

					return one;
				}
			};

			return new RandomVariableDifferentiableAbstract(result, Arrays.asList(this), partialDerivativeFunction, factory);
		}

		@Override
		public RandomVariableInterface sub(double value) {
			return add(-value);
		}

		@Override
		public RandomVariableInterface mult(double value) {
			RandomVariableInterface result = values.mult(value);

			PartialDerivativeFunction partialDerivativeFunction = new PartialDerivativeFunction(null) {	
				@Override
				public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {

					if(parameterIndex != 0) return null;

					return factory.createRandomVariableNonDifferentiable(0.0, value);
				}
			};

			return new RandomVariableDifferentiableAbstract(result, Arrays.asList(this), partialDerivativeFunction, factory);
		}

		@Override
		public RandomVariableInterface div(double value) {
			return mult(1.0/value);
		}

		@Override
		public RandomVariableInterface pow(double exponent) {
			return new RandomVariableDifferentiableAbstract(
					values.pow(exponent), 
					Arrays.asList(this),
					new PartialDerivativeFunction(values) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return X.pow(exponent - 1.0).mult(exponent);
							default:
								return null;
							}

						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface average() {
			return new RandomVariableDifferentiableAbstract(
					values.average(), 
					Arrays.asList(this),
					new PartialDerivativeFunction(null) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex) {
							return null;
						}
						
						@Override
						public RandomVariableInterface getDerivativeProduct(int parameterIndex, RandomVariableInterface derivative) {
							if(parameterIndex != 0) return null;
							return derivative.average();
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface getConditionalExpectation(
				ConditionalExpectationEstimatorInterface conditionalExpectationOperator) {			
			return new RandomVariableDifferentiableAbstract(
					values.average(), 
					Arrays.asList(this),
					new PartialDerivativeFunction(null) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex) {
							return null;
						}
						
						@Override
						public RandomVariableInterface getDerivativeProduct(int parameterIndex, RandomVariableInterface derivative) {
							if(parameterIndex != 0) return null;
							return conditionalExpectationOperator.getConditionalExpectation(derivative);
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface squared() {
			return new RandomVariableDifferentiableAbstract(
					values.squared(), 
					Arrays.asList(this),
					new PartialDerivativeFunction(this) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return X.mult(2.0);
							default:
								return null;
							}
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface sqrt() {
			return pow(0.5);
		}

		@Override
		public RandomVariableInterface exp() {
			return new RandomVariableDifferentiableAbstract(
					values.exp(), 
					Arrays.asList(this),
					new PartialDerivativeFunction(this) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return X.exp();
							default:
								return null;
							}
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface log() {
			return new RandomVariableDifferentiableAbstract(
					values.log(), 
					Arrays.asList(this),
					new PartialDerivativeFunction(this) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return X.log();
							default:
								return null;
							}
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface sin() {
			return new RandomVariableDifferentiableAbstract(
					values.sin(), 
					Arrays.asList(this),
					new PartialDerivativeFunction(this) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return X.cos();
							default:
								return null;
							}
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface cos() {
			return new RandomVariableDifferentiableAbstract(
					values.sin(), 
					Arrays.asList(this),
					new PartialDerivativeFunction(this) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return X.sin().mult(-1.0);
							default:
								return null;
							}
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface add(RandomVariableInterface randomVariable) {
			return new RandomVariableDifferentiableAbstract(
					values.add(randomVariable), 
					Arrays.asList(this, randomVariable),
					new PartialDerivativeFunction(null) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
							case 1:
								return one;
							default:
								return null;
							}
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface sub(RandomVariableInterface randomVariable) {
			return new RandomVariableDifferentiableAbstract(
					values.sub(randomVariable), 
					Arrays.asList(this, randomVariable),
					new PartialDerivativeFunction(null) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return one;
							case 1:
								return one.mult(-1.0);
							default:
								return null;
							}							
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface mult(RandomVariableInterface randomVariable) {
			return new RandomVariableDifferentiableAbstract(
					values.mult(randomVariable), 
					Arrays.asList(this, randomVariable),
					new PartialDerivativeFunction(this, randomVariable) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return Y;
							case 1:
								return X;
							default:
								return null;
							}							
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface div(RandomVariableInterface randomVariable) {
			return new RandomVariableDifferentiableAbstract(
					values.div(randomVariable), 
					Arrays.asList(this, randomVariable),
					new PartialDerivativeFunction(this, randomVariable) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return Y.invert();
							case 1:
								return X.mult(-1.0).div(Y.squared());
							default:
								return null;
							}							
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface cap(RandomVariableInterface cap) {
			
			return new RandomVariableDifferentiableAbstract(
					values.cap(cap), 
					Arrays.asList(this, cap),
					new PartialDerivativeFunction(this, cap) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return X.barrier(X.sub(Y), zero, one);
							case 1:
								return X.barrier(X.sub(Y), one, zero);
							default:
								return null;
							}							
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface floor(RandomVariableInterface floor) {
			return new RandomVariableDifferentiableAbstract(
					values.floor(floor), 
					Arrays.asList(this, floor),
					new PartialDerivativeFunction(this, floor) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return X.barrier(X.sub(Y), one, zero);
							case 1:
								return X.barrier(X.sub(Y), zero, one);
							default:
								return null;
							}							
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface accrue(RandomVariableInterface rate, double periodLength) {
			return new RandomVariableDifferentiableAbstract(
					values.accrue(rate, periodLength),
					Arrays.asList(this, rate),
					new PartialDerivativeFunction(null, rate) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return Y.mult(periodLength).add(1.0);
							case 1:
								return X.mult(periodLength);
							default:
								return null;
							}							
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface discount(RandomVariableInterface rate, double periodLength) {
			return new RandomVariableDifferentiableAbstract(
					values.discount(rate, periodLength),
					Arrays.asList(this, rate),
					new PartialDerivativeFunction(this, rate) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return Y.mult(periodLength).add(1.0).invert();
							case 1:
								return X.mult(-periodLength).div(Y.mult(periodLength).add(1.0).squared());
							default:
								return null;
							}							
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface barrier(RandomVariableInterface trigger,
				RandomVariableInterface valueIfTriggerNonNegative, RandomVariableInterface valueIfTriggerNegative) {
			// for discontinuous functions apply central finite differences
			return apply((x,y,z) -> x >= 0.0 ? y : z, valueIfTriggerNonNegative, valueIfTriggerNegative);
		}

		@Override
		public RandomVariableInterface barrier(RandomVariableInterface trigger,
				RandomVariableInterface valueIfTriggerNonNegative, double valueIfTriggerNegative) {
			// for discontinuous functions apply central finite differences
			return apply((x,y) -> x >= 0.0 ? y : valueIfTriggerNegative, valueIfTriggerNonNegative);
		}

		@Override
		public RandomVariableInterface invert() {
			return pow(-1.0);
		}

		@Override
		public RandomVariableInterface abs() {
			// for discontinuous functions apply central finite differences
			return apply(FastMath::abs);
		}

		@Override
		public RandomVariableInterface addProduct(RandomVariableInterface factor1, double factor2) {
			return new RandomVariableDifferentiableAbstract(
					values.addProduct(factor1, factor2),
					Arrays.asList(this, factor1),
					new PartialDerivativeFunction(null) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return one;
							case 1:
								return factory.createRandomVariableNonDifferentiable(0.0, factor2);
							default:
								return null;
							}							
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface addProduct(RandomVariableInterface factor1, RandomVariableInterface factor2) {
			return new RandomVariableDifferentiableAbstract(
					values.addProduct(factor1, factor2),
					Arrays.asList(this, factor1, factor2),
					new PartialDerivativeFunction(null, factor1, factor2) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return one;
							case 1:
								return Z;
							case 2:
								return Y;
							default:
								return null;
							}							
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface addRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
			return new RandomVariableDifferentiableAbstract(
					values.addRatio(numerator, denominator),
					Arrays.asList(this, numerator, denominator),
					new PartialDerivativeFunction(null, numerator, denominator) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return one;
							case 1:
								return Z.invert();
							case 2:
								return Y.mult(-1.0).div(Z.squared());
							default:
								return null;
							}							
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface subRatio(RandomVariableInterface numerator,
				RandomVariableInterface denominator) {
			return new RandomVariableDifferentiableAbstract(
					values.subRatio(numerator, denominator),
					Arrays.asList(this, numerator, denominator),
					new PartialDerivativeFunction(null, numerator, denominator) {	
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex ) {
							switch (parameterIndex) {
							case 0:
								return one;
							case 1:
								return Z.mult(-1.0).invert();
							case 2:
								return Y.div(Z.squared());
							default:
								return null;
							}							
						}
					},
					getFactory());
		}

		@Override
		public RandomVariableInterface addSumProduct(List<RandomVariableInterface> factor1,
				List<RandomVariableInterface> factor2) {
			List<RandomVariableInterface> parents = Arrays.asList(this);
			parents.addAll(factor1);
			parents.addAll(factor2);

			List<Boolean> keepValues = new ArrayList<>(parents.size());
			keepValues.replaceAll(item -> true);
			keepValues.add(0, false);			

			return new RandomVariableDifferentiableAbstract(
					addSumProduct(factor1, factor2),
					parents,
					new PartialDerivativeFunction(parents, keepValues) {

						private int numberOfFactors = factor1.size();
						@Override
						public RandomVariableInterface getPartialDerivativeFor(int parameterIndex) {
							if(parameterIndex == 0) return one;
							if(parameterIndex > numberOfFactors) return parentValues.get(parameterIndex - numberOfFactors);
							else return parentValues.get(parameterIndex + numberOfFactors);
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface isNaN() {
			return getValues().isNaN();
		}

		@Override
		public Map<Long, RandomVariableInterface> getGradient() {
			return opteratorTreeNode.getGradient();
		}

		public Map<Long, RandomVariableInterface> getAllPartialDerivatives() {
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

//		@Override
//		public String toString() {
//			return "RandomVariableDifferentiableAbstract [values=" + values + ", ID=" + getID() + "]";
//		}
	}

	/**
	 * abstract class implementing the functions to calculate the analytic partial derivative with respect to some parameter index
	 * 
	 * @author Stefan Sedlmair
	 * @version 1.0
	 * */
	private abstract static class PartialDerivativeFunction{

		protected static final RandomVariableInterface zero = new RandomVariable(0.0);
		protected static final RandomVariableInterface one = new RandomVariable(1.0);
		
		
		protected final List<RandomVariableInterface> parentValues;

		protected final RandomVariableInterface X, Y, Z;

		public PartialDerivativeFunction(RandomVariableInterface X, RandomVariableInterface Y, RandomVariableInterface Z) {
			parentValues = null;//Arrays.asList(X,Y,Z);
			this.X = RandomVariableDifferentiableAbstract.valuesOf(X);
			this.Y = RandomVariableDifferentiableAbstract.valuesOf(Y);
			this.Z = RandomVariableDifferentiableAbstract.valuesOf(Z);
		}

		public PartialDerivativeFunction(RandomVariableInterface X, RandomVariableInterface Y) {
			this(X,Y,null);
		}

		public PartialDerivativeFunction(RandomVariableInterface X) {
			this(X,null,null);
		}

		public PartialDerivativeFunction(List<RandomVariableInterface> parentValues, List<Boolean> keepValues) {
			for(int i = 0; i < parentValues.size(); i++)
				if(!keepValues.get(i)) parentValues.set(i, null);
			this.parentValues = parentValues;
			this.X = null;
			this.Y = null;
			this.Z = null;
//			this.X = (parentValues.size() > 0) ? parentValues.get(0) : null;
//			this.Y = (parentValues.size() > 1) ? parentValues.get(1) : null;
//			this.Z = (parentValues.size() > 2) ? parentValues.get(2) : null;
		}

		/**
		 * Implement the derivative function for the operation which generated the result random variable y,
		 * for a function of the form
		 * <center> y = f(x<sub>1</sub>,...,x<sub>n</sub>) </center> 
		 * 
		 * 
		 * @param parameterIndex gives the index i for x<sub>i</sub> to calculate the partial derivative to f for.
		 * @return {@link RandomVariableInterface} representing the partial derivative df(x<sub>1</sub>,...,x<sub>n</sub>)/dx<sub>i</sub>
		 * */
		public abstract RandomVariableInterface getPartialDerivativeFor(int parameterIndex);
		
		public RandomVariableInterface getDerivativeProduct(int parameterIndex, RandomVariableInterface derivative) {
			return getPartialDerivativeFor(parameterIndex).mult(derivative);
		}
	}
}
