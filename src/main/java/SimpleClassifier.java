import javafx.application.Platform;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class SimpleClassifier {
    public static void main(String[] args) throws Exception {
        int batchSize=1;   int outputSize=3;   int classIndex=4;
        double learninRate=0.001;
        int inputSize=4;    int numHiddenNodes=10;
        MultiLayerNetwork model;
        int nEpochs=100;
        InMemoryStatsStorage inMemoryStatsStorage;

        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(learninRate))
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(numHiddenNodes)
                        .activation(Activation.SIGMOID).build())
                .layer(1,new OutputLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(outputSize)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .activation(Activation.SOFTMAX).build())
                .build();
        model=new MultiLayerNetwork(configuration);
        model.init();

        UIServer uiServer=UIServer.getInstance();
        inMemoryStatsStorage=new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        //model.setListeners(new ScoreIterationListener(10));
        model.setListeners(new StatsListener(inMemoryStatsStorage));

        File fileTrain=new ClassPathResource("iris-train.csv").getFile();
        RecordReader recordReaderTrain=new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        DataSetIterator dataSetIteratorTrain=
                new RecordReaderDataSetIterator(recordReaderTrain,batchSize,classIndex,outputSize);
        for (int i = 0; i <nEpochs ; i++) {
            model.fit(dataSetIteratorTrain);
        }


        System.out.println("Model Evaluation");
        File fileTest=new ClassPathResource("irisTest.csv").getFile();
        RecordReader recordReaderTest=new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest=
                new RecordReaderDataSetIterator(recordReaderTest,batchSize,classIndex,outputSize);
        Evaluation evaluation=new Evaluation(outputSize);

        while (dataSetIteratorTest.hasNext()){
            DataSet dataSet = dataSetIteratorTest.next();
            INDArray features=dataSet.getFeatures();
            INDArray labels=dataSet.getLabels();
            INDArray predicted=model.output(features);
            evaluation.eval(labels,predicted);
        }
        System.out.println(evaluation.stats());


    }
}
