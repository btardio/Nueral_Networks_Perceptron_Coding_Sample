/*
 *    
 *
 *    Copyright (C) 2016 Brandon C Tardio
 *
 *    This file is part of NN_Perceptron
 *
 *    NN_Perceptron is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    NN_Perceptron is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with NN_Perceptron.  If not, see <http://www.gnu.org/licenses/>.
 *    
 *    Contact: BTardio@gmail.com
 *             818 424 6838
 *
 *
 *    ( NN_Perceptron )
 *
 *
 *
 */

/*
 * PTRON.java
 *
 * Created on October 5, 2005, 2:53 PM
 *
 */

import java.io.*;
import java.util.*;
import java.io.ObjectInputStream;
import java.io.InputStream;

public class PTRON {
    private int inputSize;
    private int outputSize;
    private int numTrainingInstances;
    private int weightSize;
    private int y_inSize;
    private int ySize;
    private int biasSize;
    private int trainingDataInputSize;
    private int trainingDataOutputSize;
    private int[] trainingDataInput;
    private int[] trainingDataOutput;
    private float[] weight;
    private float[] y_in;
    private int[] y;
    private float[] bias;
    private float THETA = 0;
    private HashMap outputPatterns;
    
    
    void loadWeights(String filename)throws Exception{
        
        File in = new File(filename);
        FileInputStream inputFile = new FileInputStream(in);
        StreamTokenizer st = new StreamTokenizer(inputFile);
        
        File inP = new File(filename + ".patterns");
        FileInputStream inputFileP = new FileInputStream(inP);
        StreamTokenizer stP = new StreamTokenizer(inputFileP);
        
        
        st.nextToken();
        inputSize = (int)st.nval;  
        System.out.println("inputSize: " + inputSize);
        st.nextToken();
        outputSize = (int)st.nval; 
        System.out.println("outputSize: " + outputSize);
        
        weightSize = inputSize * outputSize;
        y_inSize = outputSize;
        ySize = outputSize;
        biasSize = outputSize;
        
        weight = new float[weightSize];
        bias = new float[biasSize];
        
        for(int i = 0; i < weightSize; i++){
            st.nextToken();
            weight[i] = (float)st.nval;
        }
        
        for(int i = 0; i < biasSize; i++){
            st.nextToken();
            bias[i] = (float)st.nval;
        }
        
        
        stP.nextToken();
        int numPatterns = (int)stP.nval;
        stP.nextToken();
        outputPatterns = new HashMap();
        for( int i = 0; i < numPatterns; i++){
            int[] tempArray = new int[outputSize];
            String tempStr = new String();
            for(int q = 0; q < outputSize; q++){
                stP.nextToken();
                tempArray[q] = (int)stP.nval;
            }
            stP.nextToken();
            tempStr = stP.sval;
            
            outputPatterns.put((Object)tempArray.clone(), (Object)tempStr.toString());
            
        }
        inputFile.close();
        inputFileP.close();
        
        
    
    }
    
    void saveWeights(String filename) throws Exception{
        
        File outputFile = new File(filename);
        FileWriter out = new FileWriter(outputFile);

        File outputFileTwo = new File(filename + ".patterns");
        FileWriter outP = new FileWriter(outputFileTwo);

        
        out.write(String.valueOf(inputSize));
        out.write(" ");
        out.write(String.valueOf(outputSize));
        out.write(" ");
        
        for( int i = 0; i < weightSize; i++ ){	
            out.write(String.valueOf(weight[i]));
            out.write(" ");
        
        }
        
        // save bias
        for( int i = 0; i < biasSize; i++){
            out.write(String.valueOf(bias[i]));
            out.write(" ");
        
        }
        
        out.close();
        
        Set tempSet = outputPatterns.keySet();
        Object[] setObjects;
        setObjects = tempSet.toArray();
        
        
        outP.write(Integer.toString(setObjects.length) + " ");
        outP.write(Integer.toString(outputSize) + " ");
        
        Object[] keyValuePairs = outputPatterns.entrySet().toArray();
        
        for( int i = 0; i < outputPatterns.size(); i++){
            Map.Entry entry = (Map.Entry) keyValuePairs[i];
            int[] key = (int[])entry.getKey();
            Object value = entry.getValue();
            for( int w = 0; w < key.length; w++){
                outP.write(Integer.toString(key[w]) + " ");
            }
            outP.write((String)value + " ");
            
        }
        
        outP.close();                
    }
    
    void showWeights(){
        for(int i = 0; i < outputSize; i++){
	
            System.out.println("Printing weight for output node number " + i);
	
            for(int j = 0; j < inputSize; j++){
                System.out.println("weight[" + ((i*inputSize)+j) + "] : " + weight[(i*inputSize) + j]);
            }
	
            System.out.println("bias[" + i + "] : " + bias[i]);
        }
    }
    float calcSum(int[] x, float[] w, int offsetX, int offsetW, int n){
        float sum = 0; 
        
        for( int i = 0; i < n; i++){
            sum += x[i + offsetX] * w[i + offsetW];
        }
        return sum;
    }
    
    
    void train(){
    
        
        // STEP 0
        for(int i = 0; i < weightSize; i++){
            weight[i] = 0;
        }
        for(int i = 0; i < biasSize; i++){
            bias[i] = 0;
        }

        boolean stoppingCondition = false;
    
        boolean needToUpdateWeights = false;
    
        int bleh;
    
        // STEP 1
        // while stopping condition is false, do steps 1-6
        while(!stoppingCondition){
            
            stoppingCondition = true;
            needToUpdateWeights = false;
	
            // STEP 2
            // for each bipolar training pair s : t do steps 3-5
            for(int c = 0; c < numTrainingInstances; c++){
            
                // STEP 3
                // set activation of each input unit, i = 1,...,n
                // they are set in the readFile procedure
	    
                // STEP 4
                // compute activation of each output unit
                for( int j = 0; j < outputSize; j++){
		
                    int weightBegin = j * inputSize;
                    int tInputBegin = c * inputSize;
                    int tOutputBegin = c * outputSize;
		
                    y_in[j]= bias[j] + calcSum(trainingDataInput, 
					   weight, 
                                           tInputBegin,
                                           weightBegin,
					   inputSize);
		
		
                    if( y_in[j] > THETA ){
                        y[j] = 1;
                    }
                    else if( -THETA <= y_in[j] && y_in[j] <= THETA ){
                        y[j] = 0;
                    }
                    else{
                        y[j] = -1;
                    }
		
                    System.out.println(trainingDataOutput[tOutputBegin+j]);
                    
                    if(trainingDataOutput[tOutputBegin+j]!=y[j]){
		    
		    
		    
                        for( int i = 0; i < inputSize; i++){
			
                            stoppingCondition = false;
			
			
                            weight[weightBegin + i] = weight[weightBegin + i] + 
                                (trainingDataOutput[tOutputBegin + j] * 
                                trainingDataInput[tInputBegin + i]);
						
                        }
		    
                        bias[j] = bias[j] + trainingDataOutput[tOutputBegin+j]; 
                    }

                }

            }
        }
    
    }
    
    public void readTrainingInputFile(String filename) throws Exception{

        File in = new File(filename);
        FileInputStream inputFile = new FileInputStream(in);
        StreamTokenizer st = new StreamTokenizer(inputFile);
        
        
        String inputStr;
        
        
        st.nextToken();
        inputSize = (int)st.nval;  
        System.out.println("inputSize: " + inputSize);
        st.nextToken();
        outputSize = (int)st.nval; 
        System.out.println("outputSize: " + outputSize);
        st.nextToken();
        numTrainingInstances = (int)st.nval; 
        System.out.println("numInstances: " + numTrainingInstances);

        weightSize = inputSize * outputSize;
        y_inSize = outputSize;
        ySize = outputSize;
        biasSize = outputSize;

        trainingDataInputSize = numTrainingInstances * inputSize;
        trainingDataOutputSize = numTrainingInstances * outputSize;
 
  
        weight = new float[weightSize];
 
        y_in = new float[y_inSize];
        y = new int[ySize];
        bias = new float[biasSize];
        
        int[] tempOutputPattern;
        String tempOutputPatternString;
        outputPatterns = new HashMap();
        
        trainingDataInput = new int[trainingDataInputSize];
        trainingDataOutput = new int[trainingDataOutputSize];            
    
        
        String tempInputString;
        for( int d = 0; d < numTrainingInstances; d++){
            
            tempOutputPattern = new int[outputSize];
            tempOutputPatternString = new String();
            
            for( int j = d * inputSize; j < (d * inputSize) + inputSize; j++){
                st.nextToken();
                trainingDataInput[j] = (int)st.nval;
                System.out.println(trainingDataInput[j]);
            }
	
            for( int w = d * outputSize; w < (d * outputSize) + outputSize; w++){
                int q = w % outputSize;
                st.nextToken();
                tempOutputPattern[q] = trainingDataOutput[w] = (int)st.nval; //inputFile.readInt();
                System.out.println(trainingDataOutput[w]);
	    
            }
	
            st.nextToken();
            
            tempOutputPatternString = st.sval;
            
            outputPatterns.put((Object)tempOutputPattern.clone(), (Object)tempOutputPatternString.toString());
            
            
            	
        }
    
        
        inputFile.close();

    }

    public void deploy(String filename) throws Exception{

        File outputFile = new File(filename + ".output");
        FileWriter out = new FileWriter(outputFile);

        
        File in = new File(filename);
        FileInputStream inputFile = new FileInputStream(in);
        StreamTokenizer st = new StreamTokenizer(inputFile);
        
        y = new int[outputSize];
        
        st.nextToken();
        int tempInputSize = (int)st.nval;
        st.nextToken();
        int tempOutputSize = (int)st.nval;
        st.nextToken();
        int tempNumInstances = (int)st.nval;
    
        
        
        if( tempInputSize != inputSize || tempOutputSize != outputSize){
            System.out.println("Size mismatch.");
            return;
        }
    
        int[] dinput = new int[inputSize];
        int[] doutput = new int[outputSize];
    
        y_in = new float[outputSize];    
        
        for( int c = 0; c < tempNumInstances; c++){
	
            System.out.println("INPUT: ");
            for( int i = 0; i < inputSize; i++){
                st.nextToken();
                dinput[i] = (int)st.nval;
                out.write(Integer.toString(dinput[i]) + " ");
            }
	
            
            for ( int j = 0; j < outputSize; j++){
	    
                int weightBegin = j * inputSize;
	    
	    
                y_in[j] = bias[j] + calcSum(dinput, weight, 0, weightBegin, inputSize);
                
                
                if( y_in[j] > THETA ){
                    y[j] = 1;
                }
                else if( -THETA <= y_in[j] && y_in[j] <= THETA ){
                    y[j] = 0;
                }
                else{
                    y[j] = -1;
                }
                
                
            }
            
            // going to have to iterate the map keys manually, no idea
            // how to set up/redefine a comparison function
          
            Iterator keyValuePairs = outputPatterns.entrySet().iterator();
            
            for(int i = 0; i < outputPatterns.size(); i++){
                Map.Entry entry = (Map.Entry) keyValuePairs.next();
                Object key = entry.getKey();
                Object value = entry.getValue();
                boolean equality = true;
                for( int t = 0; t < y.length; t++){
                    if(y[t] != ((int[])key)[t])
                        equality = false;
                }
                if(equality){
                    out.write((String)value.toString() + " ");
                    
                }

            }

            
        }      
        out.close();
    }
    
}
