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
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.stream.Collectors;
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
public class RandomVariableDifferentiableFunctionalFactory extends AbstractRandomVariableDifferentiableFactory {

	private final static AtomicLong nextIdentifier = new AtomicLong(0); 

	private final double finiteDifferencesStepSize;

	private final boolean enableAD;
	private final boolean enableAAD;

	/**
	 * @param randomVariableFactoryForNonDifferentiable
	 */
	public RandomVariableDifferentiableFunctionalFactory(AbstractRandomVariableFactory randomVariableFactoryForNonDifferentiable, Map<String, Object> properties) {
		super(randomVariableFactoryForNonDifferentiable);

		// step-size for the usage of finite differences if no analytic derivative is given
		this.finiteDifferencesStepSize 	= (double) properties.getOrDefault("finiteDifferencesStepSize", 1E-8);

		// enable AD by keeping track of upwards 
		this.enableAD 				= (boolean) properties.getOrDefault("enableAD", true);
		this.enableAAD 				= (boolean) properties.getOrDefault("enableAAD", true);
	}

	public RandomVariableDifferentiableFunctionalFactory(AbstractRandomVariableFactory randomVariableFactoryForNonDifferentiable) {
		this(randomVariableFactoryForNonDifferentiable, new HashMap<>());
	}

	public RandomVariableDifferentiableFunctionalFactory() {
		this(new RandomVariableFactory(), new HashMap<>());
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableFunctionalFactory#createRandomVariable(double, double)
	 */
	@Override
	public RandomVariableDifferentiableInterface createRandomVariable(double time, double value) {
		RandomVariableInterface randomvariable = super.createRandomVariableNonDifferentiable(time, value);
		return new RandomVariableDifferentiableFunctional(randomvariable, null, null, this);
	}

	/* (non-Javadoc)
	 * @see net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableFunctionalFactory#createRandomVariable(double, double[])
	 */
	@Override
	public RandomVariableDifferentiableInterface createRandomVariable(double time, double[] values) {		
		RandomVariableInterface randomvariable = super.createRandomVariableNonDifferentiable(time, values);
		return new RandomVariableDifferentiableFunctional(randomvariable, null, null, this);
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

		private final List<OperatorTreeNode> parentTreeNodes;
		private final List<OperatorTreeNode> childTreeNodes;

		private final PartialDerivativeFunction derivatives;	

		/**
		 * Operator Tree Node that holds parent and child information according to the settings of the given {@link RandomVariableDifferentiableFunctionalFactory}.
		 * <lu>
		 * 	<li>if <code>enableAD = true</code> save upward- and downward facing dependencies</li>
		 * 	<li>if <code>enableAAD = true</code> only save downward facing dependencies</li>
		 * 	<li>else do not save any dependencies and no {@link PartialDerivativeFunction}</li>
		 * </lu>
		 * @param parents {@link List} of {@link OperatorTreeNode}s from parents   
		 * @param partialDerivativeFunction {@link PartialDerivativeFunction} to calculate the next addend for the chain rule of derivation
		 * @param factory {@link RandomVariableDifferentiableFunctionalFactory} holding the setting on what to save for the operator tree
		 * */
		public OperatorTreeNode(List<OperatorTreeNode> parents, PartialDerivativeFunction partialDerivativeFunction, RandomVariableDifferentiableFunctionalFactory factory) {
			// get identifier
			this.id = nextIdentifier.getAndIncrement();
			 
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
			if(factory.enableAD || factory.enableAAD){
				this.derivatives = partialDerivativeFunction;
				this.parentTreeNodes = parents;
			}
			// in case neither algorithmic differentiation technique is selected do not save partial derivative function either
			else{ 
				this.derivatives = null;
				this.parentTreeNodes = null;
			}
		}

		/**
		 * method implements reverse mode of algorithmic differentiation
		 * 
		 * @return {@link Map} with key id and value  dV<sub>this</sub>/dx<sub>id</sub>
		 * */
		private Map<Long, RandomVariableInterface> getGradient(Set<Long> targetIDs) {
			
			long minID = 0;
			if(targetIDs != null && !targetIDs.isEmpty()) minID = targetIDs.parallelStream().reduce(Long::min).get();
			
			Map<Long, RandomVariableInterface> gradient = Collections.synchronizedMap(new HashMap<>());

			// thread save treeMap
			ConcurrentSkipListMap<Long, OperatorTreeNode> treeNodesToPropagte = new ConcurrentSkipListMap<>();

			// partial derivative with respect to itself
			gradient.put(id, randomVariableFromConstant(1.0));

			// add id of this variable to propagate downwards
			treeNodesToPropagte.put(id, this);			

			// sequentially go through all treeNodes in the respective operator tree and propergate the derivatives downwards
			while(!treeNodesToPropagte.isEmpty()){

				// get and remove highest ID from treeNodesToPropagate
				Entry<Long, OperatorTreeNode> highestEntry = treeNodesToPropagte.pollLastEntry();

				Long childID = highestEntry.getKey();
				if(childID < minID) break;

				OperatorTreeNode childTreeNode = highestEntry.getValue();
				
				final List<OperatorTreeNode> parentTreeNodes = childTreeNode.parentTreeNodes;
								
				if(parentTreeNodes == null) continue;
				
				propergateTreeNodesThroughOperatorTree(gradient, childTreeNode, parentTreeNodes, treeNodesToPropagte);
				
				// if not defined otherwise delete child after derivative has been propagated downwards to parents
				if(targetIDs == null || !targetIDs.contains(childID)) gradient.remove(childID); 
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
			treeNodes.stream().forEach( treeNode -> {
					// if parentTreeNode is null, derivative is zero, thus continue with next treeNode
					if(treeNode == null) return;

					// get the current parent id
					long ID = treeNode.id;
					
					/* get product of partial derivative and derivative from root/leaf to origin (is different for AD and AAD) */
					RandomVariableInterface newAddendOfChainRuleSum = (originID > ID) ? 
							/*AAD*/	treeNodeOfOrigin.getDerivativeProduct(treeNode, derivative):
							/* AD*/	treeNode.getDerivativeProduct(treeNodeOfOrigin, derivative);
							
					
					// chain rule - get already existing part of the sum, if it does not exist yet start the sum with zero
					RandomVariableInterface existingChainRuleSum = derivatives.getOrDefault(ID, randomVariableFromConstant(0.0));

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
			if(derivatives == null) return randomVariableFromConstant(0.0);
						
			// map TreeNode to index
			int parameterIndex = parentTreeNodes.indexOf(treeNode); 
			
			// if parameterIndex smaller zero, this value is independent of this ID
			if(parameterIndex < 0) return randomVariableFromConstant(0.0);
			
			// calculate the multiplication for the chain rule sum
			return derivatives.getDerivativeProduct(parameterIndex, derivative);
		}

		/**
		 * method implements tangent mode of algorithmic differentiation
		 * 
		 * @return {@link Map} with key id and value  dV<sub>id</sub>/dx<sub>this</sub>
		 * */
		private Map<Long, RandomVariableInterface> getAllPartialDerivatives(Set<Long> targetIDs){
			
			long maxID = Long.MAX_VALUE;
			if(targetIDs != null && !targetIDs.isEmpty()) maxID = targetIDs.parallelStream().reduce(Long::max).get();
			
			Map<Long, RandomVariableInterface> partialDerivatives = new HashMap<>();

			// every child in the operator tree is of the same instance of its parents
			ConcurrentSkipListMap<Long, OperatorTreeNode> treeNodesToPropagte = new ConcurrentSkipListMap<>();

			// partial derivative with respect to itself
			partialDerivatives.put(id, randomVariableFromConstant(1.0));

			// add id of this variable to propagate upwards
			treeNodesToPropagte.put(id, this);

			while(!treeNodesToPropagte.isEmpty()){

				// get and remove smallest ID from treeNodesToPropagate
				Entry<Long, OperatorTreeNode> lowestEntry = treeNodesToPropagte.pollFirstEntry();

				Long parentID = lowestEntry.getKey();
				if(parentID > maxID) break;
				OperatorTreeNode parentOperatorTreeNode = lowestEntry.getValue();

				final List<OperatorTreeNode> childTreeNodes = parentOperatorTreeNode.childTreeNodes;
				
				if(childTreeNodes.isEmpty()) continue;
				
				propergateTreeNodesThroughOperatorTree(partialDerivatives, parentOperatorTreeNode, childTreeNodes, treeNodesToPropagte);
	
				// if not defined otherwise delete parent after derivative has been propagated upwards to children
				if(targetIDs == null || !targetIDs.contains(parentID)) partialDerivatives.remove(parentID); 
			}
			return partialDerivatives;
		}
		
		private static RandomVariableInterface randomVariableFromConstant(double value){
			return RandomVariableDifferentiableFunctional.randomVariableFromConstant(value);
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
	public static class RandomVariableDifferentiableFunctional implements RandomVariableDifferentiableInterface {

		private static final long serialVersionUID = 2036109523330671173L;

		private 	  RandomVariableInterface values;
		private final OperatorTreeNode opteratorTreeNode;
		private final RandomVariableDifferentiableFunctionalFactory factory;


		/**
		 * Constructor for generating a {@link RandomVariableDifferentiableFunctional} 
		 * 
		 * @param randomvariable {@link RandomVariableInterface} to store values
		 * @param parents {@link List} of {@link RandomVariableInterface}s that where arguments of the function which resulted in randomVariable (<code>null</code> if non exist)
		 * @param partialDerivativeFunction {@link PartialDerivativeFunction} defining the way the partial derivatives are calculated (if <code>null</code> partial derivative will always be zero)
		 * @param factory {@link RandomVariableDifferentiableFunctionalFactory} factory to construct new {@link RandomVariableInterface}s
		 * */
		public RandomVariableDifferentiableFunctional(RandomVariableInterface randomvariable, List<RandomVariableInterface> parents, PartialDerivativeFunction partialDerivativeFunction, 
				RandomVariableDifferentiableFunctionalFactory factory) {

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
		
		private RandomVariableDifferentiableFunctional(RandomVariableInterface randomvariable, List<RandomVariableInterface> parents,List<Boolean> keepValues,
				BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface> partialDerivativeFunction,
				RandomVariableDifferentiableFunctionalFactory factory){
			this(randomvariable, parents, new PartialDerivativeFunction(parents, keepValues, partialDerivativeFunction),factory);
		}
		
		private RandomVariableDifferentiableFunctional(RandomVariableInterface randomvariable, List<RandomVariableInterface> parents,List<Boolean> keepValues,
				BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface> partialDerivativeFunction, 
				BinaryOperator<RandomVariableInterface> derivativeProductFunction,
				RandomVariableDifferentiableFunctionalFactory factory){
			this(randomvariable, parents, new PartialDerivativeFunction(parents, keepValues, partialDerivativeFunction, derivativeProductFunction),factory);
		}
		
		/**
		 * Extracts the values argument if parameter is of instance {@link RandomVariableDifferentiableFunctional}
		 * 
		 * @param randomVariable
		 * @return values of randomVariable if instance of {@link RandomVariableDifferentiableFunctional}
		 * */
		private static RandomVariableInterface valuesOf(RandomVariableInterface randomVariable) {
			return randomVariable instanceof RandomVariableDifferentiableFunctional ? ((RandomVariableDifferentiableFunctional)randomVariable).values : randomVariable;
		}

		private static OperatorTreeNode treeNodeOf(RandomVariableInterface randomVariable) {
			return randomVariable instanceof RandomVariableDifferentiableFunctional ? ((RandomVariableDifferentiableFunctional)randomVariable).opteratorTreeNode : null;
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

		private RandomVariableDifferentiableFunctionalFactory getFactory() {
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
			return ((RandomVariableDifferentiableFunctional) getValues()).getValues().getQuantile(quantile, probabilities);
		}

		/* (non-Javadoc)
		 * @see net.finmath.stochastic.RandomVariableInterface#getQuantileExpectation(double, double)
		 */
		@Override
		public double getQuantileExpectation(double quantileStart, double quantileEnd) {
			return ((RandomVariableDifferentiableFunctional) getValues()).getValues().getQuantileExpectation(quantileStart, quantileEnd);
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
			return new RandomVariableDifferentiableFunctional(
					values.apply(operator),
					Arrays.asList(this),
					Arrays.asList(true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(0).add(epsilonX).apply(operator).sub(x.get(0).sub(epsilonX).apply(operator)).div(2.0 * epsilonX);
						default:
							return null;
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
			return new RandomVariableDifferentiableFunctional(
					values.apply(operator, argument),
					Arrays.asList(this, argument),
					Arrays.asList(true,true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(0).add(epsilonX).apply(operator, x.get(1)).sub(x.get(0).sub(epsilonX).apply(operator, x.get(1))).div(2.0 * epsilonX);
						case 1:
							return x.get(0).apply(operator, x.get(1).add(epsilonY)).sub(x.get(0).apply(operator, x.get(1).sub(epsilonY))).div(2.0 * epsilonY);
						default:
							return null;
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
			return new RandomVariableDifferentiableFunctional(
					values.apply(operator, argument1, argument2),
					Arrays.asList(this, argument1, argument2),
					Arrays.asList(true,true, true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(0).add(epsilonX).apply(operator, x.get(1), x.get(2)).sub(x.get(0).sub(epsilonX).apply(operator, x.get(1), x.get(2))).div(2.0 * epsilonX);
						case 1:
							return x.get(0).apply(operator, x.get(1).add(epsilonY), x.get(2)).sub(x.get(0).apply(operator, x.get(1).sub(epsilonY), x.get(2))).div(2.0 * epsilonY);
						case 2:
							return x.get(0).apply(operator, x.get(1), x.get(2).add(epsilonZ)).sub(x.get(0).apply(operator, x.get(1), x.get(2).sub(epsilonZ))).div(2.0 * epsilonZ);
						default:
							return null;
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface floor(double floor) {
			return floor(randomVariableFromConstant(floor));
		}

		@Override
		public RandomVariableInterface cap(double cap) {
			return cap(randomVariableFromConstant(cap));
		}

		@Override
		public RandomVariableInterface add(double value) {
			return new RandomVariableDifferentiableFunctional(
					values.add(value),
					Arrays.asList(this),
					Arrays.asList(false),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return randomVariableFromConstant(1.0);
						default:
							return null;
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface sub(double value) {
			return add(-value);
		}

		@Override
		public RandomVariableInterface mult(double value) {
			return new RandomVariableDifferentiableFunctional(
					values.mult(value),
					Arrays.asList(this),
					Arrays.asList(false),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return randomVariableFromConstant(value);
						default:
							return null;
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface div(double value) {
			return mult(1.0/value);
		}

		@Override
		public RandomVariableInterface pow(double exponent) {
			return new RandomVariableDifferentiableFunctional(
					values.pow(exponent),
					Arrays.asList(this),
					Arrays.asList(true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(0).pow(exponent - 1.0).mult(exponent);
						default:
							return null;
						}
					}, 
					getFactory());
			
		}

		@Override
		public RandomVariableInterface average() {
			return new RandomVariableDifferentiableFunctional(
					values.average(),
					Arrays.asList(this),
					Arrays.asList(false),
					null, 
					(BinaryOperator<RandomVariableInterface>) (x,y) -> y.average(),
					getFactory());
		}

		@Override
		public RandomVariableInterface getConditionalExpectation(
				ConditionalExpectationEstimatorInterface conditionalExpectationOperator) {	
			return new RandomVariableDifferentiableFunctional(
					values.average(),
					Arrays.asList(this),
					Arrays.asList(false),
					null, 
					(BinaryOperator<RandomVariableInterface>) (x,y) -> conditionalExpectationOperator.getConditionalExpectation(y),
					getFactory());
		}

		@Override
		public RandomVariableInterface squared() {
			return new RandomVariableDifferentiableFunctional(
					values.squared(),
					Arrays.asList(this),
					Arrays.asList(true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(0).mult(2.0);
						default:
							return null;
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
			return new RandomVariableDifferentiableFunctional(
					values.exp(),
					Arrays.asList(this),
					Arrays.asList(true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(0).exp();
						default:
							return null;
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface log() {
			return new RandomVariableDifferentiableFunctional(
					values.log(),
					Arrays.asList(this),
					Arrays.asList(true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(0).invert();
						default:
							return null;
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface sin() {
			return new RandomVariableDifferentiableFunctional(
					values.sin(),
					Arrays.asList(this),
					Arrays.asList(true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(0).cos();
						default:
							return null;
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface cos() {
			return new RandomVariableDifferentiableFunctional(
					values.cos(),
					Arrays.asList(this),
					Arrays.asList(true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(0).sin().mult(-1.0);
						default:
							return null;
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface add(RandomVariableInterface randomVariable) {
			return new RandomVariableDifferentiableFunctional(
					values.add(randomVariable),
					Arrays.asList(this, randomVariable),
					Arrays.asList(false, false),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
						case 1:
							return randomVariableFromConstant(1.0);
						default:
							return null;
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface sub(RandomVariableInterface randomVariable) {
			return new RandomVariableDifferentiableFunctional(
					values.sub(randomVariable),
					Arrays.asList(this, randomVariable),
					Arrays.asList(false, false),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return randomVariableFromConstant(1.0);
						case 1:
							return randomVariableFromConstant(-1.0);
						default:
							return null;
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface mult(RandomVariableInterface randomVariable) {
			return new RandomVariableDifferentiableFunctional(
					values.mult(randomVariable),
					Arrays.asList(this, randomVariable),
					Arrays.asList(true, true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(1);
						case 1:
							return x.get(0);
						default:
							return null;
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface div(RandomVariableInterface randomVariable) {
			return new RandomVariableDifferentiableFunctional(
					values.div(randomVariable),
					Arrays.asList(this, randomVariable),
					Arrays.asList(true, true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(1).invert();
						case 1:
							return x.get(0).mult(-1.0).div(x.get(1).squared());
						default:
							return null;
						}
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface cap(RandomVariableInterface cap) {
			return new RandomVariableDifferentiableFunctional(
					values.cap(cap),
					Arrays.asList(this, cap),
					Arrays.asList(true, true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(0).barrier(x.get(0).sub(x.get(1)), randomVariableFromConstant(0.0), randomVariableFromConstant(1.0));
						case 1:
							return x.get(0).barrier(x.get(0).sub(x.get(1)), randomVariableFromConstant(1.0), randomVariableFromConstant(0.0));
						default:
							return null;
						}		
					}, 
					getFactory());								
		}

		@Override
		public RandomVariableInterface floor(RandomVariableInterface floor) {
			return new RandomVariableDifferentiableFunctional(
					values.floor(floor),
					Arrays.asList(this, floor),
					Arrays.asList(true, true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(0).barrier(x.get(0).sub(x.get(1)), randomVariableFromConstant(1.0), randomVariableFromConstant(0.0));
						case 1:
							return x.get(0).barrier(x.get(0).sub(x.get(1)), randomVariableFromConstant(0.0), randomVariableFromConstant(1.0));
						default:
							return null;
						}		
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface accrue(RandomVariableInterface rate, double periodLength) {
			return new RandomVariableDifferentiableFunctional(
					values.accrue(rate, periodLength),
					Arrays.asList(this, rate),
					Arrays.asList(true, true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(1).mult(periodLength).add(1.0);
						case 1:
							return x.get(0).mult(periodLength);
						default:
							return null;
						}		
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface discount(RandomVariableInterface rate, double periodLength) {
			return new RandomVariableDifferentiableFunctional(
					values.discount(rate, periodLength),
					Arrays.asList(this, rate),
					Arrays.asList(true, true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return x.get(1).mult(periodLength).add(1.0).invert();
						case 1:
							return x.get(0).mult(-periodLength).div(x.get(1).mult(periodLength).add(1.0).squared());
						default:
							return null;
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
			return new RandomVariableDifferentiableFunctional(
					values.addProduct(factor1, factor2),
					Arrays.asList(this, factor1),
					Arrays.asList(false, false),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return randomVariableFromConstant(1.0);
						case 1:
							return randomVariableFromConstant(factor2);
						default:
							return null;
						}		
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface addProduct(RandomVariableInterface factor1, RandomVariableInterface factor2) {
			return new RandomVariableDifferentiableFunctional(
					values.addProduct(factor1, factor2),
					Arrays.asList(this, factor1, factor2),
					Arrays.asList(false, true, true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return randomVariableFromConstant(1.0);
						case 1:
							return x.get(2);
						case 2:
							return x.get(1);
						default:
							return null;
						}		
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface addRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
			return new RandomVariableDifferentiableFunctional(
					values.addRatio(numerator, denominator),
					Arrays.asList(this, numerator, denominator),
					Arrays.asList(false, true, true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return randomVariableFromConstant(1.0);
						case 1:
							return x.get(2).invert();
						case 2:
							return x.get(1).div(x.get(2).squared()).mult(-1.0);
						default:
							return null;
						}		
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface subRatio(RandomVariableInterface numerator,
				RandomVariableInterface denominator) {
			return new RandomVariableDifferentiableFunctional(
					values.subRatio(numerator, denominator),
					Arrays.asList(this, numerator, denominator),
					Arrays.asList(false, true, true),
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						switch (i) {
						case 0:
							return randomVariableFromConstant(1.0);
						case 1:
							return x.get(2).invert().mult(-1.0);
						case 2:
							return x.get(1).div(x.get(2).squared());
						default:
							return null;
						}		
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface addSumProduct(List<RandomVariableInterface> factor1,
				List<RandomVariableInterface> factor2) {
			List<RandomVariableInterface> parents = new ArrayList<>();
			parents.add(this);
			parents.addAll(factor1);
			parents.addAll(factor2);

			List<Boolean> keepValues = new ArrayList<>();
			keepValues.add(false);
			for(int i = 0; i < parents.size()-1;i++) keepValues.add(true);

			return new RandomVariableDifferentiableFunctional(
					values.addSumProduct(factor1, factor2),
					parents,
					keepValues,
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x, i) -> {
						int numberOfFactors = (int) ((x.size() - 1) / 2);
						if(i == 0) return randomVariableFromConstant(1.0);
						if(i > numberOfFactors) return x.get(i - numberOfFactors);
						else return x.get(i + numberOfFactors);
					}, 
					getFactory());
		}

		@Override
		public RandomVariableInterface isNaN() {
			return getValues().isNaN();
		}

		@Override
		public Map<Long, RandomVariableInterface> getGradient(Set<Long> targetIDs) {
			if(!factory.enableAD && !factory.enableAAD) throw new UnsupportedOperationException();
			return opteratorTreeNode.getGradient(targetIDs);
		}

		public Map<Long, RandomVariableInterface> getAllPartialDerivatives(Set<Long> targetIDs) {
			if(!factory.enableAD) throw new UnsupportedOperationException();
			return opteratorTreeNode.getAllPartialDerivatives(targetIDs);
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

		@Override
		public String toString() {
			return "RandomVariableDifferentiableFunctional [values=" + values.toString() + ", ID=" + getID() + "]";
		}
		
		private static RandomVariableInterface randomVariableFromConstant(double value){
			return new RandomVariable(value);
		}
	}

	/**
	 * class implementing the functions to calculate the analytic partial derivative with respect to some parameter index
	 * 
	 * @author Stefan Sedlmair
	 * @version 1.0
	 * */
	private static class PartialDerivativeFunction{
		
		final BinaryOperator<RandomVariableInterface> derivativeProduct;
		final BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface> partialDerivativeFunction;
		
		private final List<RandomVariableInterface> parentValues;

		public PartialDerivativeFunction(List<RandomVariableInterface> parentValues, List<Boolean> keepValues ,
				BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface> partialDerivativeFunction,
				BinaryOperator<RandomVariableInterface> derivativeProduct) {
			int numberOfParents = parentValues.size();
			this.parentValues = IntStream.range(0, numberOfParents).mapToObj(
					i -> keepValues.get(i) ? RandomVariableDifferentiableFunctional.valuesOf(parentValues.get(i)) : null
							).collect(Collectors.toList());
			
			this.partialDerivativeFunction = (partialDerivativeFunction == null) ? 
					(BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface>) (x,y) -> null 
					: partialDerivativeFunction;
			this.derivativeProduct = derivativeProduct;
		}
		
		public PartialDerivativeFunction(List<RandomVariableInterface> parentValues, List<Boolean> keepValues ,
				BiFunction<List<RandomVariableInterface>, Integer, RandomVariableInterface> partialDerivativeFunction) {
			this(parentValues, keepValues, partialDerivativeFunction, (x,y) -> x.mult(y));
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
		public RandomVariableInterface getPartialDerivativeFor(int parameterIndex){
			return partialDerivativeFunction.apply(parentValues, parameterIndex);
		}
		
		public RandomVariableInterface getDerivativeProduct(int parameterIndex, RandomVariableInterface derivative) {
			return derivativeProduct.apply(getPartialDerivativeFor(parameterIndex), derivative);
		}
	}
}
