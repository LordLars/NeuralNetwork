package Main;

import AI.*;

import java.util.Arrays;

public class Main {

    private float[][] inputs;
    private Action[] targets;

    private Action[] outputOptions;

    private int inputLength;
    private int outputLength;
    private int miniBatchLength;



    public Main(){
        miniBatchLength = 10;
        inputLength = 5;
        outputLength = 5;
        inputs = new float[miniBatchLength][];
        targets = new Action[miniBatchLength];
        outputOptions = new Action[]{Action._1,Action._2,Action._3,Action._4,Action._5};
        int[] layers = new int[]{inputLength,10,10,outputLength};

        NN nn = new NN(layers, outputOptions, ActivationType.ReLU, .01f, miniBatchLength,0.9f);


        long startTime = System.currentTimeMillis();
        for(int i = 0; i < 1000;i++){
            setInput();
            nn.propagation(inputs);
            nn.backPropagation(targets);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Finished: " + (endTime - startTime) + "ms");

        System.out.println(Arrays.toString(nn.propagation(new float[][]{{1,0,0,0,0}})));
        System.out.println(nn.getCurrentAccuracy());
        System.out.println(nn.getOutput());
    }

    private void setInput(){
        for(int j = 0; j < miniBatchLength;j++){
            inputs[j] = new float[inputLength];
            int index = (int)(Math.random()*inputLength);
            inputs[j][index] = 1f;
            targets[j] = outputOptions[4-index];
        }
    }
}
