/**
 * 
 */
package net.finmath.functions;

/**
 * @author Stefan Sedlmair
 * @version 1.0
 */
public class VectorAlgbra {
	
	public static void sameLength(double[] x, double[] y){
		if(x.length != y.length) throw new IllegalArgumentException("Vectors are not equal in size!");
	}
	
	public static int numberOfRows(double[][] A){
		return A[0].length;
	}

	public static int numberOfColumns(double[][] A){
		return A.length;
	}
	
	public static double[][] transpose(double[][] A){
		double[][] transposeA = new double[numberOfRows(A)][numberOfColumns(A)];
		for(int i = 0; i < numberOfRows(A); i++)
			for(int j = 0; j < numberOfColumns(A); j++)
				transposeA[i][j] = A[j][i];
		return transposeA;
	}
	
	public static double[] dotProduct(double[][] A, double[] x){
		if(numberOfColumns(A) != x.length) throw new IllegalArgumentException("Dimension mismatch, cannot multiply vector to matrix!");
		double[] y = new double[numberOfRows(A)];
		for(int i=0; i < y.length; i++)
			y[i] = innerProduct(A[i], x);
		return y;
	}
	
	public static double[] dotProduct(double[] x, double[][] A){
		return dotProduct(transpose(A), x);
	}
	
	public static double innerProduct(double[] x, double[] y){
		sameLength(x, y);
		double dotProduct = 0;
		for(int i=0; i < x.length; i++)
			dotProduct += x[i]*y[i];
		return dotProduct;
	}
	
	public static double[][] outerProduct(double[] x, double[] y){
		double[][] outerProduct = new double[x.length][y.length];
		for(int i=0; i < x.length; i++)
			for(int j = 0; j < y.length; j++)
				outerProduct[i][j] = x[i]*y[j];
		return outerProduct;
	}
	
	public static double[] add(double[] x, double[] y){
		sameLength(x, y);
		double[] z = x.clone();
		for(int i=0; i < x.length; i++)
			z[i] += y[i];
		return z;
	}
	
	public static double[] subtract(double[] x, double[] y){
		sameLength(x, y);
		double[] z = x.clone();
		for(int i=0; i < x.length; i++)
			z[i] -= y[i];
		return z;
	}
	
	public static double[] hadamardProduct(double[] x, double[] y){
		sameLength(x, y);
		double[] z = x.clone();
		for(int i=0; i < x.length; i++)
			z[i] *= y[i];
		return z;
	}

	public static double[] scalarProduct(double a, double[] x){
		double[] y = x.clone();
		for(int i=0; i < y.length; i++)
			y[i] *= a;
		return y;
	}
	
	public static boolean containsNaN(double[] X){
		for(double x : X) if(Double.isNaN(x)) return true;
		return false;
	}
	
	public static boolean isNaN(double[] X){
		for(double x : X) if(!Double.isNaN(x)) return false;
		return true;
	}	
}
