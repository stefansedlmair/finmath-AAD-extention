package net.finmath.montecarlo.automaticdifferentiation.forward;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import net.finmath.montecarlo.RandomVariable;
import net.finmath.montecarlo.automaticdifferentiation.AbstractRandomVariableDifferentiableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.montecarlo.automaticdifferentiation.backward.RandomVariableDifferentiableAADFactory;
import net.finmath.stochastic.RandomVariableInterface;

@RunWith(Parameterized.class)
public class RandomVariableADTest {


	/* parameters specify the factories one wants to test against each other */
	@Parameters(name="{1}")
	public static Collection<Object[]> data(){
		
		List<Object[]> config = new ArrayList<>();
		
		config.add(new Object[]{ new RandomVariableADFactory(), "AD"});
		config.add(new Object[]{ new RandomVariableDifferentiableAADFactory(), "AAD"});

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

		RandomVariable randomVariable01 = new RandomVariable(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0});
		RandomVariable randomVariable02 = new RandomVariable(0.0,
				new double[] {-4.0, -2.0, 0.0, 2.0, 4.0} );

		/*x_1*/
		RandomVariableInterface aadRandomVariable01 = randomVariableFactory.createRandomVariable(randomVariable01.getFiltrationTime(), randomVariable01.getRealizations());

		/*x_2*/
		RandomVariableInterface aadRandomVariable02 =  randomVariableFactory.createRandomVariable(randomVariable02.getFiltrationTime(), randomVariable02.getRealizations());


		/* x_3 = x_1 + x_2 */
		RandomVariableInterface aadRandomVariable03 = aadRandomVariable01.add(aadRandomVariable02);
		/* x_4 = x_3 * x_1 */
		RandomVariableInterface aadRandomVariable04 = aadRandomVariable03.mult(aadRandomVariable01);
		/* x_5 = x_4 + x_1 = ((x_1 + x_2) * x_1) + x_1 = x_1^2 + x_2x_1 + x_1*/
		RandomVariableInterface aadRandomVariable05 = aadRandomVariable04.add(aadRandomVariable01);

		Map<Long, RandomVariableInterface> aadGradient = ((RandomVariableDifferentiableInterface)aadRandomVariable05).getGradient();

		/* dy/dx_1 = x_1 * 2 + x_2 + 1
		 * dy/dx_2 = x_1 */
		RandomVariableInterface[] analyticGradient = new RandomVariableInterface[]{
				randomVariable01.mult(2.0).add(randomVariable02).add(1.0),
				randomVariable01
		};

		Long[] keys = new Long[aadGradient.keySet().size()];
		keys = aadGradient.keySet().toArray(keys);
		Arrays.sort(keys);

		for(int i=0; i<analyticGradient.length;i++){
			Assert.assertTrue(analyticGradient[i].equals(aadGradient.get(keys[i])));
		}
	}

	@Test
	public void testRandomVariableSimpleGradient2(){

		RandomVariable randomVariable01 = new RandomVariable(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0});
		RandomVariable randomVariable02 = new RandomVariable(0.0,
				new double[] {-4.0, -2.0, 0.0, 2.0, 4.0} );

		/*x_1*/
		RandomVariableInterface aadRandomVariable00 = randomVariableFactory.createRandomVariable(randomVariable01.getFiltrationTime(), randomVariable01.getRealizations());

		/*x_2*/
		RandomVariableInterface aadRandomVariable01 = randomVariableFactory.createRandomVariable(randomVariable02.getFiltrationTime(), randomVariable02.getRealizations());

		/* x_3 = x_1 + x_2 */
		RandomVariableInterface aadRandomVariable02 = aadRandomVariable00.add(aadRandomVariable01);
		/* x_4 = x_3 * x_1 */
		RandomVariableInterface aadRandomVariable03 = aadRandomVariable02.mult(aadRandomVariable00);
		/* x_5 = x_4 + x_1 = ((x_1 + x_2) * x_1) + x_1 = x_1^2 + x_2x_1 + x_1*/
		RandomVariableInterface aadRandomVariable04 = aadRandomVariable03.add(aadRandomVariable00);

		Map<Long, RandomVariableInterface> aadGradient = ((RandomVariableDifferentiableInterface) aadRandomVariable04).getGradient();

		/* dy/dx_1 = x_1 * 2 + x_2 + 1
		 * dy/dx_2 = x_1 */
		RandomVariableInterface[] analyticGradient = new RandomVariableInterface[]{
				randomVariable01.mult(2.0).add(randomVariable02).add(1.0),
				randomVariable01
		};

		Long[] keys = new Long[aadGradient.keySet().size()];
		keys = aadGradient.keySet().toArray(keys);
		Arrays.sort(keys);

		for(int i=0; i<analyticGradient.length;i++){
			Assert.assertTrue(analyticGradient[i].equals(aadGradient.get(keys[i])));
		}
	}
}
