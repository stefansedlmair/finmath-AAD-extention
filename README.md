# finmath-lib algorithmic differentiation extentions

This project provides and interface (<code>RandomVariableDifferentiableInterface</code>)
for random variables which provide algorithmic differentiation. The interfaces
extends RandomVariableInterface and hence allows to use algorithmic differentiation 
techniques in all Monte-Carlo contexts (via a replacement of the corresponding 
parameters / factories).

The project provides an implementation of the forward (a.k.a. tangent) and
the backward (a.k.a. adjoint) method via RandomVariableDifferentiableFunctionalFactory.

The interface RandomVariableDifferentiableInterface will introduce
three major additional methods:

	Long getID();	
	Map<Long, RandomVariableInterface> getGradient();
	Map<Long, RandomVariableInterface> getAllPartialDerivatives();

The method <code>getGradient</code> will return a map providing the first order
differentiation of the given random variable (this) with respect to
*all* its input <code>RandomVariableDifferentiableInterface</code>s (root tree-nodes). To get the differentiation with respect to a specific object use

	Map gradient = X.getGradient();
	RandomVariableInterface derivative = X.get(Y.getID());
	
The method <code>getAllPartialDerivatives</code> works accordingly but returns a map containing the partial derivatives for *all* its dependent <code>RandomVariableDifferentiableInterface</code>s (leaf tree-nodes) with respect to the given random variable (this).
