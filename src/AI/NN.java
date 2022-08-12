package AI;

import java.io.*;
import java.util.Arrays;

public class NN {

    private NNLayer[] layers;
    private final Action[] oOptions;

    private float[] output;
    private float accuracy;
    private int guesses;
    private int rightGuesses;
    private final int[] layerSizes;

    private final ActivationFunction activationFunction;
    private final float nLernRate;

    public NN(int[] layerSizes, Action[] oOptions, ActivationType activation, float nLernRate) {
        this.layerSizes = layerSizes;
        this.oOptions = oOptions;
        this.nLernRate = nLernRate;
        activationFunction = new ActivationFunction(activation);
        buildLayers();
    }

    //region LAYER BUILDER
    /**
     * Builds Layers
     */
    public void buildLayers() {
        layers = new NNLayer[layerSizes.length-1];
        for(int i = 0; i < layers.length; i++) {
            layers[i] = new NNLayer(layerSizes[i], layerSizes[i+1],activationFunction, nLernRate);
        }
    }
    //endregion

    //region PROPAGATION
    /**
     * Propagation Method
     */
    public float[] propagation(float[] inputs) {
        for(int i = 0; i < layers.length-1; i++){
            inputs = layers[i].propagation(inputs);
        }
        output = layers[layers.length-1].propagation(inputs);
        return output;
    }
    //endregion

    //region BACKPROPAGATION
    /**
     * Backpropagation method
     * @param target targets of the AI
     */
    public void backPropagation(Action target) {
        float[] targets = new float[output.length];
        int maxIndex = 0;
        for(int i = 1; i < output.length;i++) if(output[i] > output[maxIndex]) maxIndex = i;
        updateAccuracy(target,maxIndex);
        targets[Arrays.asList(oOptions).indexOf(target)] = 1;
        for(int i = 0; i < output.length;i++) targets[i] = 2*(output[i] - targets[i]);

        layers[layers.length - 1].backPropagation(targets);
        for(int i = layers.length - 2; i >= 0; i--) layers[i].backPropagation(layers[i + 1].getNextChain());
    }

    /**
     * Updates the Accuracy
     * @param target targets of the AI
     */
    private void updateAccuracy(Action target, int maxIndex){
        guesses += 1;
        if(oOptions[maxIndex].equals(target)) rightGuesses += 1;
        accuracy = (float) rightGuesses / guesses * 100;
    }

    //endregion

    //region SAVE & LOAD

    /**
     * Saves the AI
     */
    public void save(String path) {
        if(accuracy > getSavedAccuracy(path)){
            try {
                FileOutputStream fos = new FileOutputStream(path);
                BufferedOutputStream bf = new BufferedOutputStream(fos);
                ObjectOutputStream obj = new ObjectOutputStream(bf);
                NNData data = new NNData();
                NNLayerData[] layerData = new NNLayerData[layers.length];
                for(int i = 0; i < layers.length; i++) layerData[i] = layers[i].save();
                data.nnLayerData = layerData;
                data.accuracy = accuracy;
                obj.writeObject(data);
                obj.close();
            } catch (IOException ignored) {}
        }
    }

    /**
     * Loads the AI
     */
    public void load(String path) {
        try {
            FileInputStream fis = new FileInputStream(path);
            BufferedInputStream bis = new BufferedInputStream(fis);
            ObjectInputStream ois = new ObjectInputStream(bis);
            NNData data = (NNData) ois.readObject();
            for(int i = 0; i < layers.length; i++) layers[i].load(data.nnLayerData[i]);
        } catch (IOException | ClassNotFoundException ignored){}
    }

    //endregion

    //region GETTER & SETTER
    /**
     * @return Saved accuracy
     */
    public float getSavedAccuracy(String path){
        try {
            FileInputStream fis = new FileInputStream(path);
            BufferedInputStream bis = new BufferedInputStream(fis);
            ObjectInputStream ois = new ObjectInputStream(bis);

            NNData data = (NNData) ois.readObject();
            return data.accuracy;

        } catch (IOException | ClassNotFoundException ignored) {}
        return 0;
    }

    /**
     * @return Current accuracy
     */
    public float getCurrentAccuracy(){
        return accuracy;
    }

    //endregion
}
