package net.finmath.montecarlo.interestrate.modelplugins;

import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretizationInterface;

public class LIBORVolatilityModelFourParameterExponentialForm extends LIBORVolatilityModel {

	private RandomVariableInterface[] parameters;
    
    /**
     * Creates the volatility model &sigma;<sub>i</sub>(t<sub>j</sub>) = ( a + b * (T<sub>i</sub>-t<sub>j</sub>) ) * exp(-c (T<sub>i</sub>-t<sub>j</sub>)) + d
     * 
     * @param timeDiscretization The simulation time discretization t<sub>j</sub>.
     * @param liborPeriodDiscretization The period time discretization T<sub>i</sub>.
     * @param a The parameter a: an initial volatility level.
     * @param b The parameter b: the slope at the short end (shortly before maturity).
     * @param c The parameter c: exponential decay of the volatility in time-to-maturity.
     * @param d The parameter d: if c &gt; 0 this is the very long term volatility level.
     * @param isCalibrateable Set this to true, if the parameters are available for calibration.
     */
    public LIBORVolatilityModelFourParameterExponentialForm(
    		TimeDiscretizationInterface timeDiscretization, 
    		TimeDiscretizationInterface liborPeriodDiscretization, 
    		RandomVariableInterface 	a,
    		RandomVariableInterface 	b,
    		RandomVariableInterface 	c,
    		RandomVariableInterface 	d){
    	this(timeDiscretization, liborPeriodDiscretization, new RandomVariableInterface[]{a,b,c,d});
    }
    
    public LIBORVolatilityModelFourParameterExponentialForm(
    		TimeDiscretizationInterface timeDiscretization, 
    		TimeDiscretizationInterface liborPeriodDiscretization, 
    		RandomVariableInterface[] 	parameters) {
        super(timeDiscretization, liborPeriodDiscretization);
        this.parameters = parameters;
    }
   

    /* (non-Javadoc)
     * @see net.finmath.montecarlo.interestrate.modelplugins.LIBORVolatilityModel#getVolatility(int, int)
     */
    @Override
    public RandomVariableInterface getVolatility(int timeIndex, int liborIndex) {
        // Create a very simple volatility model here
        double time             = getTimeDiscretization().getTime(timeIndex);
        double maturity         = getLiborPeriodDiscretization().getTime(liborIndex);
        double timeToMaturity   = maturity-time;

        RandomVariableInterface volatilityInstanteaneous; 
        if(timeToMaturity <= 0)
        {
        	/*to get a zero random variable of same instance as $a$ first cap $a$ at zero and later floor it at zero */
            volatilityInstanteaneous = parameters[0].cap(0.0);   // This forward rate is already fixed, no volatility
        }
        else
        {
            volatilityInstanteaneous = parameters[0].add(parameters[1]).mult(timeToMaturity).mult(parameters[2].mult(-1.0 * timeToMaturity).exp()).add(parameters[3]);
        }

        return volatilityInstanteaneous.floor(0.0);
    }

	@Override
	public Object clone() {
		return new LIBORVolatilityModelFourParameterExponentialForm(
				super.getTimeDiscretization(),
				super.getLiborPeriodDiscretization(),
				parameters
				);
	}

	@Override
	public RandomVariableInterface[] getParameter() {
		return parameters;
	}

	@Override
	public void setParameter(RandomVariableInterface[] parameters) {
		if(parameters == null) return;
		if(parameters.length != 4) throw new IllegalArgumentException();
		
		this.parameters = parameters;
	}


}
