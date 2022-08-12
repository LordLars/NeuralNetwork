package Main;

import AI.*;

import java.util.Arrays;

public class Main {

    public Main(){
        float[] input = new float[]{1,1,0,0,1};
        Action[] output = new Action[]{Action.Zero,Action.One};
        int[] layers = new int[]{input.length,10,10,output.length};
        NN nn = new NN(layers, output, ActivationType.ReLU, .001f);

        long startTime = System.currentTimeMillis();
        for(int i = 0; i < 1_000_000;i++){
            nn.propagation(input);
            nn.backPropagation(Action.One);
        }
        System.out.println(Arrays.toString(nn.propagation(input)));
        System.out.println(nn.getCurrentAccuracy());
        long endTime = System.currentTimeMillis();
        System.out.println("Finished: " + (endTime - startTime) + "ms");

    }
}
