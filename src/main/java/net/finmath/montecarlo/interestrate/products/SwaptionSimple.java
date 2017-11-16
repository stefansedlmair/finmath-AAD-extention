/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 15.02.2004
 */
package net.finmath.montecarlo.interestrate.products;

import java.util.function.DoubleUnaryOperator;

import net.finmath.exception.CalculationException;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.marketdata.model.curves.DiscountCurveInterface;
import net.finmath.marketdata.model.curves.ForwardCurveInterface;
import net.finmath.marketdata.products.Swap;
import net.finmath.marketdata.products.SwapAnnuity;
import net.finmath.montecarlo.interestrate.LIBORModelMonteCarloSimulationInterface;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.RegularSchedule;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationInterface;

/**
 * Implements the valuation of a simplified (idealized) swaption under a
 * LIBORModelMonteCarloSimulationInterface
 * 
 * @author Christian Fries
 * @version 1.2
 */
public class SwaptionSimple extends AbstractLIBORMonteCarloProduct {

	public enum ValueUnit {
		VALUE,
		INTEGRATEDLOGNORMALVARIANCE,
		INTEGRATEDNORMALVARIANCE,
		INTEGRATEDVARIANCE,	/// Backward compatibility, same as INTEGRATEDLOGNORMALVARIANCE
		VOLATILITYLOGNORMAL,
		VOLATILITYNORMAL,
		VOLATILITY	/// Backward compatibility, same as VOLATILITY_LOGNORMAL
	}

	private final TimeDiscretizationInterface	tenor;
	private final double						swaprate;
	private final Swaption						swaption;
	private final ValueUnit						valueUnit;

	/**
	 * Note: It is implicitly assumed that swapTenor[0] is the exercise date (no forward starting).
	 * @param swaprate The strike swaprate of the swaption.
	 * @param swapTenor The swap tenor in doubles.
	 */
	public SwaptionSimple(double swaprate, TimeDiscretizationInterface swapTenor) {
		this(swaprate, swapTenor.getAsDoubleArray(), ValueUnit.VALUE);
	}

	/**
	 * Note: It is implicitly assumed that swapTenor[0] is the exercise date (no forward starting).
	 * @param swaprate The strike swaprate of the swaption.
	 * @param swapTenor The swap tenor in doubles.
	 * @param valueUnit See <code>getValue(AbstractLIBORMarketModel model)</code>
	 */
	public SwaptionSimple(double swaprate, double[] swapTenor, ValueUnit valueUnit) {
		super();
		this.tenor = new TimeDiscretization(swapTenor);
		this.swaprate = swaprate;
		this.swaption	= new Swaption(swapTenor[0], tenor, swaprate);
		this.valueUnit	= valueUnit;
	}

	/**
	 * This method returns the value random variable of the product within the specified model, evaluated at a given evalutationTime.
	 * Note: For a lattice this is often the value conditional to evalutationTime, for a Monte-Carlo simulation this is the (sum of) value discounted to evaluation time.
	 * Cashflows prior evaluationTime are not considered.
	 * 
	 * @param evaluationTime The time on which this products value should be observed.
	 * @param model The model used to price the product.
	 * @return The random variable representing the value of the product discounted to evaluation time
	 * @throws net.finmath.exception.CalculationException Thrown if the valuation fails, specific cause may be available via the <code>cause()</code> method.
	 */
	@Override
	public RandomVariableInterface getValue(double evaluationTime, LIBORModelMonteCarloSimulationInterface model) throws CalculationException {
		RandomVariableInterface value = swaption.getValue(evaluationTime, model);

		if(valueUnit == ValueUnit.VALUE) return value;

		ForwardCurveInterface forwardCurve	 = model.getModel().getForwardRateCurve();
		DiscountCurveInterface discountCurve = model.getModel().getAnalyticModel() != null ? model.getModel().getAnalyticModel().getDiscountCurve(forwardCurve.getDiscountCurveName()) : null;

		double parSwaprate = Swap.getForwardSwapRate(new RegularSchedule(tenor), new RegularSchedule(tenor), forwardCurve, model.getModel().getAnalyticModel());
		double optionMaturity = tenor.getTime(0);
		double strikeSwaprate = swaprate;
		double swapAnnuity = discountCurve != null ? SwapAnnuity.getSwapAnnuity(tenor, discountCurve) : SwapAnnuity.getSwapAnnuity(tenor, forwardCurve);

		DoubleUnaryOperator impliedBlackScholesVolaFunction = x -> AnalyticFormulas.blackScholesOptionImpliedVolatility(parSwaprate, optionMaturity, strikeSwaprate, swapAnnuity, x);
		DoubleUnaryOperator impliedBachelierVolaFunction 	= x -> AnalyticFormulas.bachelierOptionImpliedVolatility(	parSwaprate, optionMaturity, strikeSwaprate, swapAnnuity, x);
		
		if(valueUnit == ValueUnit.VOLATILITY || valueUnit == ValueUnit.VOLATILITYLOGNORMAL) {
			return value.average().apply(impliedBlackScholesVolaFunction);
		}
		else if(valueUnit == ValueUnit.VOLATILITYNORMAL) {
			return value.average().apply(impliedBachelierVolaFunction);
		}
		else if(valueUnit == ValueUnit.INTEGRATEDVARIANCE  || valueUnit == ValueUnit.INTEGRATEDLOGNORMALVARIANCE) {
			return value.average().apply(impliedBlackScholesVolaFunction).squared().mult(optionMaturity);
		}
		else if(valueUnit == ValueUnit.INTEGRATEDNORMALVARIANCE) {
			return value.average().apply(impliedBachelierVolaFunction).squared().mult(optionMaturity);
		}
		else {
			throw new UnsupportedOperationException("Provided valueUnit not implemented.");
		}
	}

	@Override
	public String toString() {
		return "SwaptionSimple [tenor=" + tenor + ", swaprate=" + swaprate
				+ ", valueUnit=" + valueUnit + "]";
	}
}
