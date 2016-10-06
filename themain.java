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
 * Main.java
 *
 * Created on October 3, 2005, 8:41 PM
 *
 */

import java.io.*;
import java.util.*;

public class themain{
    
        
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        PTRON nn = new PTRON();
        
        boolean wantsToQuit = false;
    
        BufferedReader input = new BufferedReader( new InputStreamReader(System.in));
        
        String buffer = new String();
        
        while(!wantsToQuit){
	
            System.out.print("Welcome to my first neural network - Perceptron!\n");
	
            System.out.print("Choices:\n");
            System.out.print("\t1-Train using a training input data file.\n");
            System.out.print("\t2-Load weight.\n");
            System.out.print("\t3-Save weights.\n");
            System.out.print("\t4-Deploy.\n");
            System.out.print("\t5-Show Weights.\n");
            System.out.print("\t6-Quit.\n");
	
            buffer = input.readLine();

	switch(Integer.parseInt(buffer)){
            case 1:
                System.out.print("Please enter a filename:\n>");
                buffer = input.readLine();
                nn.readTrainingInputFile(buffer);
                nn.train();
                break;
            case 2:
                System.out.print("Please enter a filename:\n>");
                buffer = input.readLine();
                nn.loadWeights(buffer);
                break;
            case 3:
                System.out.print("Please enter a filename:\n>");
                buffer = input.readLine();
                nn.saveWeights(buffer);
                break;
            case 4:
                System.out.print("Please enter a filename:\n>");
                buffer = input.readLine();
                nn.deploy(buffer);
                break;
            case 5:
                nn.showWeights();
                break;
            case 6:
                wantsToQuit = true;
                break;
	    
            default:
                System.out.print("Invalid Selection\n");
                break;
            }
        }
    }
    
}


