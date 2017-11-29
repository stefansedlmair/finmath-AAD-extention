package net.finmath.montecarlo.automaticdifferentiation.forward;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import net.finmath.montecarlo.RandomVariable;
import net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableFunctionalFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.randomnumbers.MersenneTwister;
import net.finmath.stochastic.RandomVariableInterface;

@RunWith(Parameterized.class)
public class RandomVariableADTest {


	/* parameters specify the factories one wants to test against each other */
	@Parameters(name="{1}")
	public static Collection<Object[]> data(){
		
		List<Object[]> config = new ArrayList<>();
		
		config.add(new Object[]{ new RandomVariableADFactory(), "AD"});
		config.add(new Object[]{ new RandomVariableDifferentiableFactory(), "AlgorthmicDifferentiation"});
		config.add(new Object[]{ new RandomVariableDifferentiableFunctionalFactory(), "AlgorthmicDifferentiationAbstractDerivative"});

		return config;
	}

	private final AbstractRandomVariableDifferentiableFactory randomVariableFactory;

	public RandomVariableADTest(AbstractRandomVariableDifferentiableFactory factory, String name) {
		this.randomVariableFactory = factory;
	}

	@Test
	public void testRandomVariableDeterministc() {

		// Create a random variable with a constant
		RandomVariableInterface randomVariable = randomVariableFactory.createRandomVariable(2.0);

		// Perform some calculations
		randomVariable = randomVariable.mult(2.0);
		randomVariable = randomVariable.add(1.0);
		randomVariable = randomVariable.squared();
		randomVariable = randomVariable.sub(4.0);
		randomVariable = randomVariable.div(7.0);

		// The random variable has average value 3.0 (it is constant 3.0)
		Assert.assertTrue(randomVariable.getAverage() == 3.0);

		// Since the random variable is deterministic, it has zero variance
		Assert.assertTrue(randomVariable.getVariance() == 0.0);

	}

	@Test
	public void testRandomVariableStochastic() {

		// Create a stochastic random variable
		RandomVariableInterface randomVariable2 = randomVariableFactory.createRandomVariable(0.0,
				new double[] {-4.0, -2.0, 0.0, 2.0, 4.0} );

		// Perform some calculations
		randomVariable2 = randomVariable2.add(4.0);
		randomVariable2 = randomVariable2.div(2.0);

		// The random variable has average value 2.0
		Assert.assertTrue(randomVariable2.getAverage() == 2.0);

		// The random variable has variance value 2.0 = (4 + 1 + 0 + 1 + 4) / 5
		Assert.assertEquals(2.0, randomVariable2.getVariance(), 1E-12);

		// Multiply two random variables, this will expand the receiver to a stochastic one
		RandomVariableInterface randomVariable = randomVariableFactory.createRandomVariable(3.0);
		randomVariable = randomVariable.mult(randomVariable2);

		// The random variable has average value 6.0
		Assert.assertTrue(randomVariable.getAverage() == 6.0);

		// The random variable has variance value 2 * 9
		Assert.assertTrue(randomVariable.getVariance() == 2.0 * 9.0);
	}

	@Test
	public void testRandomVariableArithmeticSqrtPow() {

		// Create a stochastic random variable
		RandomVariableInterface randomVariable = randomVariableFactory.createRandomVariable(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		RandomVariableInterface check = randomVariable.sqrt().sub(randomVariable.pow(0.5));

		// The random variable is identical 0.0
		Assert.assertTrue(check.getAverage() == 0.0);
		Assert.assertTrue(check.getVariance() == 0.0);

	}

	@Test
	public void testRandomVariableArithmeticSquaredPow() {

		// Create a stochastic random variable
		RandomVariableInterface randomVariable = randomVariableFactory.createRandomVariable(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		RandomVariableInterface check = randomVariable.squared().sub(randomVariable.pow(2.0));

		// The random variable is identical 0.0
		Assert.assertTrue(check.getAverage() == 0.0);
		Assert.assertTrue(check.getVariance() == 0.0);

	}

	@Test
	public void testRandomVariableStandardDeviation() {

		// Create a stochastic random variable
		RandomVariableInterface randomVariable = randomVariableFactory.createRandomVariable(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		double check = randomVariable.getStandardDeviation() - Math.sqrt(randomVariable.getVariance());
		Assert.assertTrue(check == 0.0);
	}

	@Test
	public void testRandomVariableSimpleGradient(){

		RandomVariable x0 = new RandomVariable(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0});
		RandomVariable x1 = new RandomVariable(0.0,
				new double[] {-4.0, -2.0, 0.0, 2.0, 4.0} );

		/*x_0*/
		RandomVariableInterface randomVariable00 = randomVariableFactory.createRandomVariable(x0.getFiltrationTime(), x0.getRealizations());

		/*x_1*/
		RandomVariableInterface randomVariable01 =  randomVariableFactory.createRandomVariable(x1.getFiltrationTime(), x1.getRealizations());


		/* x_2 = x_0 + x_1 */
		RandomVariableInterface randomVariable02 = randomVariable00.add(randomVariable01);
		/* x_3 = x_2 * x_0 */
		RandomVariableInterface randomVariable03 = randomVariable02.mult(randomVariable00);
		/* x_4 = x_3 + x_0 = ((x_0 + x_1) * x_0) + x_0 = x_0^2 + x_1x_0 + x_0*/
		RandomVariableInterface randomVariable04 = randomVariable03.add(randomVariable00);

		Map<Long, RandomVariableInterface> adGradient = new HashMap<>();
		
		RandomVariableDifferentiableInterface adRandomVariable00 = (RandomVariableDifferentiableInterface) randomVariable00;
		RandomVariableDifferentiableInterface adRandomVariable01 = (RandomVariableDifferentiableInterface) randomVariable01;
		RandomVariableDifferentiableInterface adRandomVariable04 = (RandomVariableDifferentiableInterface) randomVariable04;
		
		Map<Long, RandomVariableInterface> adDerivatives00 = adRandomVariable00.getAllPartialDerivatives();
		Map<Long, RandomVariableInterface> adDerivatives01 = adRandomVariable01.getAllPartialDerivatives();
		
		adGradient.put(adRandomVariable00.getID(), adDerivatives00.get(adRandomVariable04.getID()));
		adGradient.put(adRandomVariable01.getID(), adDerivatives01.get(adRandomVariable04.getID()));	

		/* dx_4/dx_0 = x_0 * 2 + x_1 + 1
		 * dx_4/dx_1 = x_0 */
		RandomVariableInterface[] analyticGradient = new RandomVariableInterface[]{
				x0.mult(2.0).add(x1).add(1.0),
				x0
		};

		Long[] keys = new Long[adGradient.keySet().size()];
		keys = adGradient.keySet().toArray(keys);
		Arrays.sort(keys);

		for(int i=0; i<analyticGradient.length;i++){
//			System.out.println(analyticGradient[i]);
//			System.out.println(adGradient.get(keys[i]));
			Assert.assertTrue(analyticGradient[i].equals(adGradient.get(keys[i])));
		}
	}

	@Test
	public void testRandomVariableSimpleGradient2(){

		RandomVariable x0 = new RandomVariable(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0});
		RandomVariable x1 = new RandomVariable(0.0,
				new double[] {-4.0, -2.0, 0.0, 2.0, 4.0} );

		/*x_0*/
		RandomVariableInterface randomVariable00 = randomVariableFactory.createRandomVariable(x0.getFiltrationTime(), x0.getRealizations());

		/*x_1*/
		RandomVariableInterface randomVariable01 = randomVariableFactory.createRandomVariable(x1.getFiltrationTime(), x1.getRealizations());

		/* x_2 = x_0 + x_1 */
		RandomVariableInterface randomVariable02 = randomVariable00.add(randomVariable01);
		/* x_3 = x_2 * x_0 */
		RandomVariableInterface randomVariable03 = randomVariable02.mult(randomVariable00);
		/* x_4 = x_3 + x_0 = ((x_0 + x_1) * x_0) + x_0 = x_0^2 + x_1x_0 + x_0*/
		RandomVariableInterface randomVariable04 = randomVariable03.add(randomVariable00);

		Map<Long, RandomVariableInterface> adGradient = new HashMap<>();
		
		RandomVariableDifferentiableInterface adRandomVariable00 = (RandomVariableDifferentiableInterface) randomVariable00;
		RandomVariableDifferentiableInterface adRandomVariable01 = (RandomVariableDifferentiableInterface) randomVariable01;
		RandomVariableDifferentiableInterface adRandomVariable04 = (RandomVariableDifferentiableInterface) randomVariable04;
		
		adGradient.put(adRandomVariable00.getID(), adRandomVariable00.getAllPartialDerivatives().get(adRandomVariable04.getID()));
		adGradient.put(adRandomVariable01.getID(), adRandomVariable01.getAllPartialDerivatives().get(adRandomVariable04.getID()));	

		/* dy/dx_0 = x_0 * 2 + x_1 + 1
		 * dy/dx_1 = x_0 */
		RandomVariableInterface[] analyticGradient = new RandomVariableInterface[]{
				x0.mult(2.0).add(x1).add(1.0),
				x0
		};

		Long[] keys = new Long[adGradient.keySet().size()];
		keys = adGradient.keySet().toArray(keys);
		Arrays.sort(keys);

		for(int i=0; i<analyticGradient.length;i++){
//			System.out.println(analyticGradient[i]);
//			System.out.println(adGradient.get(keys[i]));
			Assert.assertTrue(analyticGradient[i].equals(adGradient.get(keys[i])));
		}
	}
	
	@Test
	public void testRandomVariableDifferentiableInterfaceMultipleIndependentFunctions() {
		
		int seed = 1234;
		MersenneTwister mt = new MersenneTwister(seed);
		
		int numberOfRalisations = (int) 1E3;
		int numberOfFunctionRepetitions = (int) 1E3;
		
		double[] realizations = new double[numberOfRalisations];
		for(int i=0; i< numberOfRalisations; i++)
			realizations[i] = mt.nextDouble();
		
		RandomVariableInterface x = randomVariableFactory.createRandomVariable(0.0, realizations);
		
		long[] resultIDs = new long[numberOfFunctionRepetitions];
		
		for(int i = 0; i < numberOfFunctionRepetitions; i++) {
			RandomVariableInterface y = x.squared().exp().mult(0.5);
			resultIDs[i] = ((RandomVariableDifferentiableInterface) y).getID();
		}
		
		long start = System.currentTimeMillis();
		Map<Long, RandomVariableInterface> partialDerivatives = ((RandomVariableDifferentiableInterface) x).getAllPartialDerivatives();
		long end = System.currentTimeMillis();
		
		System.out.println(((end-start)/1E3));
		
		RandomVariableInterface analyticResult = x.mult(x.squared().exp());
		for(int i=0; i < numberOfFunctionRepetitions; i++) {
			Assert.assertTrue(partialDerivatives.get(resultIDs[i]).equals(analyticResult));
		}
	}
}
