/**
 * 
 */
package net.finmath.functions;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * @author Stefan Sedlmair
 * @version 1.0
 */
public class VectorAlgbra {
	
	/**
	 * Check if two arrays have the same length
	 * */
	public static void sameLength(double[] x, double[] y){
		if(x.length != y.length) throw new IllegalArgumentException("Vectors are not equal in size!");
	}
	
	
	public static int numberOfRows(double[][] A){
		return A[0].length;
	}

	public static int numberOfColumns(double[][] A){
		return A.length;
	}
	
	/**
	 * Transpose a two dimensional array in the sense that 
	 * 
	 * <center>A[i][j] = A<sup>T</sup>[j][i], for all i = 1,...,n and j=1,...,m </center>
	 * where A is a n x m Matrix.
	 * 
	 * @param A matrix
	 * @return  A<sup>T</sup> transpose of A
	 * */
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
	
	public static double[][] dotProduct(double[][] A, double[][] B){
		if(numberOfRows(A) != numberOfColumns(B)) throw new IllegalArgumentException("Dimension missmatch");
		
		double[][] transposeA = transpose(A);
		
		double[][] C = new double[numberOfColumns(A)][numberOfRows(B)];
		for(int i = 0; i < numberOfColumns(B); i++)
			C[i] = dotProduct(transposeA, B[i]);
		
		return C;	
	}
	
	public static double innerProduct(double[] x, double[] y){
		sameLength(x, y);
		return IntStream.range(0, x.length).mapToDouble(i -> x[i] * y[i]).sum();
	}
	
	public static double normL2(double[] x){
		return Math.sqrt(innerProduct(x, x));
	}
	
	public static double[][] outerProduct(double[] x, double[] y){
		double[][] outerProduct = new double[x.length][y.length];
		IntStream.range(0, y.length)
		        .forEach(i -> Arrays.parallelSetAll(outerProduct[i], j -> x[j] * y[i]));
		return outerProduct;
	}
	
	public static double[] add(double[] x, double[] y){
		sameLength(x, y);
		double[] z = new double[x.length];
		Arrays.parallelSetAll(z, i -> x[i] + y[i]);
		return z;
	}
	
	public static double[] subtract(double[] x, double[] y){
		sameLength(x, y);
		double[] z = new double[x.length];
		Arrays.parallelSetAll(z, i -> x[i] - y[i]);
		return z;
	}
	
	public static double[][] add(double[][] A, double[][] B){
		if(numberOfColumns(A) != numberOfColumns(B) || numberOfRows(A) != numberOfRows(B)) 
			throw new IllegalArgumentException("Dimension mismatch!");

		double[][] res = new double[numberOfRows(A)][numberOfColumns(A)];
		IntStream.range(0, numberOfRows(A))
		        .forEach(i -> Arrays.parallelSetAll(res[i], j -> A[i][j] + B[i][j]));
		return res;

	}
		
	public static double[][] subtract(double[][] A, double[][] B){
		return add(A, scalarProduct(B, -1.0));
	}
	
	public static double[][] scalarProduct(double[][] matrix, double multiplicator){
		double[][] matrixClone = matrix.clone();
		for(int i=0; i < numberOfRows(matrixClone); i++)
			matrixClone[i] = scalarProduct(multiplicator, matrixClone[i]);
		return matrixClone;
	}
	
	public static double[][] scalarDivision(double[][] matrix, double divisor){
		return scalarProduct(matrix, 1.0/divisor);
	}
	
	public static double[] hadamardProduct(double[] x, double[] y){
		sameLength(x, y);
		double[] z = new double[x.length];
		Arrays.parallelSetAll(z, i -> x[i] * y[i]);
		return z;
	}

	public static double[] scalarProduct(double scalar, double[] vector){
		return Arrays.stream(vector.clone()).map(x -> x*scalar).toArray();
	}
	
	public static boolean containsNaN(double[] X){
		for(double x : X) if(Double.isNaN(x)) return true;
		return false;
	}
	
	public static boolean isNaN(double[] X){
		for(double x : X) if(!Double.isNaN(x)) return false;
		return true;
	}	
	
	public static boolean isAllEntriesEqual(double[] X){
		if(X != null && X.length > 1){
			double x0 = X[0];
			for(double x : X) 
				if(x != x0) return false;
		}
		return true;
	}
	
	public static double[][] getDiagonalMatrix(double[] diagonalEntries){
		int n = diagonalEntries.length;
		// double array gets initialized with zeros anyway!
		double[][] diagMatrix = new double[n][n];
		for(int i = 0; i < n; i++)
			diagMatrix[i][i] = diagonalEntries[i];
		return diagMatrix;
	}
	
	public static double[][] getDiagonalMatrix(double diagEntry, int matrixDimension){
		double[] diagEntries = new double[matrixDimension];
		Arrays.fill(diagEntries, diagEntry);
		return getDiagonalMatrix(diagEntries);
	}
	
	public static double getAverage(double[] vector){
		return Arrays.stream(vector).average().getAsDouble();
	}
}
