/**
 * 
 */
package net.finmath.functions;

import java.io.FileWriter;

/**
 * @author Stefan Sedlmair
 *
 */
public class FileManagement {

	public static FileWriter fileWriter;
	
	public static void writeOnCSVInitialize(Boolean continueWriting, String CSVName){
		try {fileWriter = new FileWriter(CSVName, continueWriting);} catch (Exception e) {System.out.println("Error in CsvFileWriter"); e.printStackTrace();}}
	public static void writeOnCSV(String a){
		try {fileWriter.append(a);} catch (Exception e) {System.out.println("Error in CsvFileWriter"); e.printStackTrace();}}
	public static void writeOnCSVnextCell(){
		try {fileWriter.append(";");} catch (Exception e) {System.out.println("Error in CsvFileWriter"); e.printStackTrace();}}
	public static void writeOnCSVnextLine(){
		try {fileWriter.append("\n");} catch (Exception e) {System.out.println("Error in CsvFileWriter"); e.printStackTrace();}}
	public static void writeOnCSVflushclose(){
		try {fileWriter.flush(); fileWriter.close();} catch (Exception e) {System.out.println("Error in CsvFileWriter"); e.printStackTrace();}}


}
