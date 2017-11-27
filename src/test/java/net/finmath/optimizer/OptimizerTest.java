package net.finmath.optimizer;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Locale;
import java.util.Map;
import java.util.stream.IntStream;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableFactory;
import net.finmath.montecarlo.automaticdifferentiation.RandomVariableDifferentiableInterface;
import net.finmath.optimizer.OptimizerFactory.OptimizerType;
import net.finmath.optimizer.OptimizerInterfaceAAD.DerivativeFunction;
import net.finmath.randomnumbers.MersenneTwister;
import net.finmath.stochastic.RandomVariableInterface;

@RunWith(Parameterized.class)
public class OptimizerTest {

	OptimizerFactory optimizerFactory;
	int numberOfParameters;
	private RandomVariableInterface x;
	private RandomVariableInterface y;
	private RandomVariableInterface[] initialParamters;
	private RandomVariableDifferentiableFactory randomVariableFactory;
	
	private static DecimalFormat formatterValue	= new DecimalFormat(" #0.00000;-#0.00000", new DecimalFormatSymbols(Locale.ENGLISH));
	private static DecimalFormat formatterDeviation	= new DecimalFormat(" 0.00000E00;-0.00000E00", new DecimalFormatSymbols(Locale.ENGLISH));

	@Parameters(name="{0}")
	public static Collection<Object[]> data() {
		Collection<Object[]> config = new ArrayList<>();
		
		config.add(new Object[] {OptimizerType.Levenberg});
		config.add(new Object[] {OptimizerType.SimpleGradientDescent});
		config.add(new Object[] {OptimizerType.GradientDescentArmijo});
		config.add(new Object[] {OptimizerType.TruncatedGaussNetwon});
		config.add(new Object[] {OptimizerType.BroydenFletcherGoldfarbShanno});
		
		return config;
	}
	
	
	public OptimizerTest(OptimizerType optimizerType) {
		
		int maxIterations = 5000;
		double errorTolerance = 0.0;
				
		this.optimizerFactory = new OptimizerFactory(optimizerType, maxIterations, errorTolerance);
		this.randomVariableFactory = new RandomVariableDifferentiableFactory();
		
		//-----------------------------------------------------------------------------------------
		
		MersenneTwister random = new MersenneTwister(2132);
		
		int numberOfDataPoints = (int) 125;
		
		double maxRange = 2;
		
		double[] valuesX = IntStream.range(0, numberOfDataPoints).parallel().mapToDouble(i ->  maxRange * 2.0 * (random.nextDouble() - 0.5)).sorted().toArray();
//		double[] valuesY = IntStream.range(0, numberOfDataPoints).parallel().mapToDouble(i ->  maxRange * 2.0 * (random.nextDouble() - 0.5)).toArray();

		this.x = randomVariableFactory.createRandomVariable(0.0, valuesX);
//		this.y = randomVariableFactory.createRandomVariable(0.0, valuesY);

		this.y = x.pow(3);
		
		//----------------------------------------------------------------------------------------
		
		int numberOfParameters = (int) 10;
		this.initialParamters = IntStream.range(0, numberOfParameters).mapToObj(i -> randomVariableFactory.createRandomVariable(0.0)).toArray(RandomVariableInterface[]::new);
	}
	
	
	@Test
	public void OneDimensionalCurveFitting() throws SolverException{
		
		final RandomVariableInterface x1 = x;
		final RandomVariableInterface y1 = y;
		
		DerivativeFunction objectiveFunction = new DerivativeFunction() {
			
			private RandomVariableInterface zero = randomVariableFactory.createRandomVariable(0.0);
			
			double[] parameterStorage = null;
			RandomVariableInterface polynomialValue;
			RandomVariableInterface[] parameter;
			
			private void updateValues(double[] parameters){
				// cache for certain parameters
				if(Arrays.equals(parameters, parameterStorage)) return;
				
				parameter = Arrays.stream(parameters).mapToObj(param -> randomVariableFactory.createRandomVariable(param)).toArray(RandomVariableInterface[]::new);
				RandomVariableInterface polynomialValues = polynomial(x1, parameter);
				polynomialValue = polynomialValues.sub(y1).squared().average().sqrt();
			}
			
			@Override
			public void setValues(double[] parameters, double[] values) throws SolverException {
				updateValues(parameters);
				// L2 norm squared
				System.arraycopy(polynomialValue.getRealizations(), 0, values, 0, 1);
			}
			
			@Override
			public void setDerivatives(double[] parameters, double[][] derivatives) throws SolverException {
				updateValues(parameters);
				Map<Long, RandomVariableInterface> gradient = ((RandomVariableDifferentiableInterface) polynomialValue).getGradient();
				double[] derivative = Arrays.stream(parameter).mapToDouble(param -> 
					gradient.getOrDefault(((RandomVariableDifferentiableInterface) param).getID(), zero).getAverage()).toArray();
				
				// fill in the calculated gradient in the derivative matrix
				for(int parameterIndex = 0; parameterIndex < parameters.length; parameterIndex++) 
					derivatives[parameterIndex][0] = derivative[parameterIndex];
			
			}
		};
		// create optimizer from factory
		double[] initialDoubles = Arrays.stream(initialParamters).mapToDouble(rv -> rv.doubleValue()).toArray();
		OptimizerInterface optimizer = optimizerFactory.getOptimizer(objectiveFunction, initialDoubles, new double[]{0.0});
		optimizer.run();
		
		// test calibration with best fit
		System.out.println();
		System.out.println(Arrays.toString(optimizer.getBestFitParameters()));
		double[] bestDoubles = optimizer.getBestFitParameters();
		RandomVariableInterface[] bestParameter = Arrays.stream(bestDoubles).mapToObj(param -> randomVariableFactory.createRandomVariable(param)).toArray(RandomVariableInterface[]::new);
		RandomVariableInterface modelValues = polynomial(x1, bestParameter);
		
		double deviationSum = 0.0;
		double deviationSquaredSum = 0.0;
		System.out.println();
		System.out.println("x" + "\t\t" + "model" + "\t\t" + "target" +  "\t\t" + "error");
		for(int i = 0; i< x.size(); i++){
			double valueModel = modelValues.get(i);
			double valueTarget = y.get(i);
			
			double error = valueModel-valueTarget;
			deviationSum += error;
			deviationSquaredSum += error*error;		
			
			System.out.println(formatterValue.format(x.get(i)) + "\t" + formatterValue.format(valueModel) + "\t" + formatterValue.format(valueTarget) + "\t" + formatterDeviation.format(error));
		}
		
		double errorRMS = Math.sqrt(deviationSquaredSum/x.size());
		double errorAverage = deviationSum/x.size();

		System.out.println();
		System.out.println("Average Error....................." + formatterDeviation.format(errorAverage));
		System.out.println("RMS Error........................." + formatterDeviation.format(errorRMS));
		System.out.println("__________________________________________________________________________________________\n");
		
		Assert.assertEquals(0.0, errorAverage, 0.2);
	}
	
	private RandomVariableInterface polynomial(RandomVariableInterface x ,RandomVariableInterface[] parameter){
		return IntStream.range(0, parameter.length).mapToObj(i -> x.pow(i).mult(parameter[i])).reduce(RandomVariableInterface::add).get();
	}

}
