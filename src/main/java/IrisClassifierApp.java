import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.property.SimpleObjectProperty;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.web.WebEngine;
import javafx.scene.web.WebView;
import javafx.stage.Stage;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.loss.MeanSquaredErrorLoss;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class IrisClassifierApp extends Application {
    int batchSize=1;   int outputSize=3;   int classIndex=4; double learninRate=0.001;
    int inputSize=4;    int numHiddenNodes=10;
    MultiLayerNetwork model;
    int nEpochs=100;
    private double progressValue=0;
    InMemoryStatsStorage inMemoryStatsStorage;
    String[] featuresLabels=new String[]{"SepalLength","SepalWidth","PetalLength","PetalWidth"};
    String [] labels=new String[]{"Iris-setosa","Iris-versicolor","Iris-virginica"};
    Collection<Map<String,Object>> trainingDataSet=new ArrayList<>();
    Collection<Map<String,Object>> testDataSet=new ArrayList<>();

    public static void main(String[] args) throws Exception {
        launch();
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        BorderPane borderPane=new BorderPane();
        HBox hBoxTop=new HBox(10); hBoxTop.setPadding(new Insets(10));
        Button buttonCreateModel=new Button("Create Model");
        Button buttonLoadData=new Button("Load data");
        Button buttonTrain=new Button("Train Model");
        Button buttonEval=new Button("Evaluate Model");
        Button buttonPrection=new Button("Predict");
        Button buttonSaveModel=new Button("Save Model");
        Button buttonLoadModel=new Button("Load Model");
        hBoxTop.getChildren().addAll(buttonCreateModel,buttonLoadData,buttonTrain,buttonEval,buttonPrection,buttonSaveModel,buttonLoadModel);
        borderPane.setTop(hBoxTop);
        Image image=new Image(new FileInputStream(new ClassPathResource("images/model.png").getFile()));
        ImageView imageView=new ImageView(image);


        TabPane tabPane=new TabPane();
       WebView webView=new WebView();
        WebEngine webEngine=webView.getEngine();
        Tab tabWebView=new Tab("Web View");
        tabWebView.setContent(webView);
        tabPane.getTabs().add(tabWebView);

        TableView<Map<String,Object>> tableViewData=new TableView<>();

        Tab tabData=new Tab("Input Data");
        BorderPane borderPane2=new BorderPane();
        HBox hBox2=new HBox(10);hBox2.setPadding(new Insets(10));
        Button buttonDataTrain=new Button("Train Data");
        Button buttonDataTest=new Button("Test Data");
        hBox2.getChildren().addAll(buttonDataTrain,buttonDataTest);
        borderPane2.setTop(hBox2);
        borderPane2.setCenter(tableViewData);
        tabData.setContent(borderPane2);
        tabPane.getTabs().add(tabData);

        Tab tabConsole=new Tab("Console");
        TextArea textAreaConsole=new TextArea();
        tabConsole.setContent(textAreaConsole);
        tabPane.getTabs().add(tabConsole);

        Tab tabPredictions=new Tab("Predictions");
        HBox hBoxPrediction=new HBox(10);hBoxPrediction.setPadding(new Insets(10));
        GridPane gridPane=new GridPane();gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);gridPane.setVgap(10);
        Label labelSepalLength=new Label("Sepal Length:");
        TextField textFieldSepalLength=new TextField("5.0");
        Label labelSepalWidth=new Label("Sepal Length:");
        TextField textFieldSepalWidth=new TextField("3.5");
        Label labelPetalLength=new Label("Petal Length:");
        TextField textFieldPetalLength=new TextField("1.3");
        Label labelPetalWidth=new Label("Petal Length:");
        TextField textFieldPetalWidth=new TextField("0.3");
        Button buttonPredict=new Button("Predict");
        Label labelPrection=new Label("?");
        gridPane.add(labelSepalLength,0,0); gridPane.add(textFieldSepalLength,1,0);
        gridPane.add(labelSepalWidth,0,1); gridPane.add(textFieldSepalWidth,1,1);
        gridPane.add(labelPetalLength,0,2); gridPane.add(textFieldPetalLength,1,2);
        gridPane.add(labelPetalWidth,0,3); gridPane.add(textFieldPetalWidth,1,3);
        gridPane.add(buttonPredict,0,4);
        gridPane.add(labelPrection,0,5);
        Image imagePrediction=new Image(new FileInputStream(new ClassPathResource("images/unknown.png").getFile()));
        ImageView imageViewPrediction=new ImageView(imagePrediction);
        hBoxPrediction.getChildren().add(gridPane);
        hBoxPrediction.getChildren().add(imageViewPrediction);
        tabPredictions.setContent(hBoxPrediction);
        tabPane.getTabs().add(tabPredictions);


        HBox hBoxCenter=new HBox(10);hBoxCenter.setPadding(new Insets(10));

        VBox vBox1=new VBox(10);vBox1.setPadding(new Insets(10));
        ProgressBar progressBar=new ProgressBar();
        progressBar.setPrefWidth(image.getWidth());
        progressBar.setProgress(0);
        vBox1.getChildren().add(progressBar);
        vBox1.getChildren().add(imageView);
        vBox1.setVisible(false);
        hBoxCenter.getChildren().add(vBox1);
        hBoxCenter.getChildren().add(tabPane);
        borderPane.setCenter(hBoxCenter);
        Scene scene=new Scene(borderPane,800,600);
        primaryStage.setScene(scene);
        primaryStage.show();

        buttonCreateModel.setOnAction(evr->{
            new Thread(()->{
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
                imageView.setVisible(true);
            }).start();
        });

        buttonLoadData.setOnAction(evt->{

            NumberFormat numberFormat=new DecimalFormat("#0.00");
            for (int i = 0; i < featuresLabels.length; i++) {
                TableColumn<Map<String,Object>,String> column=new TableColumn<>(featuresLabels[i]);
                column.setCellValueFactory(p->{
                    return new SimpleObjectProperty(p.getValue().get(p.getTableColumn().getText()));
                });
                column.setCellFactory(p->{
                    TableCell tableCell=new TableCell(){
                        @Override
                        protected void updateItem(Object item, boolean empty) {
                            super.updateItem(item, empty);
                            if(item instanceof Number)
                            this.setText(numberFormat.format(item));
                            else if(item==null)
                                this.setText("NULL");
                            else  this.setText(item.toString());
                        }
                    };
                    return tableCell;
                });
                tableViewData.getColumns().add(column);
            }
            TableColumn<Map<String,Object>,String> tableColumnLabel=new TableColumn<>("Label");
            tableColumnLabel.setCellValueFactory(p->{
                return new SimpleObjectProperty(p.getValue().get(p.getTableColumn().getText()));
            });
            tableViewData.getColumns().add(tableColumnLabel);
            viewDataSet("iris-train.csv",tableViewData);
        });
        buttonTrain.setOnAction(evt->{
           new Thread(()->{
               try {
                   progressBar.setProgress(0);
                   progressValue=0;
                   UIServer uiServer=UIServer.getInstance();
                   inMemoryStatsStorage=new InMemoryStatsStorage();
                   uiServer.attach(inMemoryStatsStorage);
                   //model.setListeners(new ScoreIterationListener(10));
                   model.setListeners(new StatsListener(inMemoryStatsStorage));


                   Platform.runLater(()->{
                       vBox1.setVisible(true);
                       webEngine.load("http://localhost:9000");

                   });
                   File fileTrain=new ClassPathResource("iris-train.csv").getFile();
                   RecordReader recordReaderTrain=new CSVRecordReader();
                   recordReaderTrain.initialize(new FileSplit(fileTrain));
                   DataSetIterator dataSetIteratorTrain=
                           new RecordReaderDataSetIterator(recordReaderTrain,batchSize,classIndex,outputSize);
               for (int i = 0; i <nEpochs ; i++) {
                   model.fit(dataSetIteratorTrain);
                   Platform.runLater(()->{
                       progressValue+=(1.0/nEpochs);
                       progressBar.setProgress(progressValue);
                   });
               }
               } catch (Exception e) {
                   e.printStackTrace();
               }
           }) .start();

        });
        buttonEval.setOnAction(evt->{
            new Thread(()->{
                try {
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
                    textAreaConsole.appendText(evaluation.stats());
                } catch (IOException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }).start();

        });

        buttonDataTrain.setOnAction(evt->{
            viewDataSet("iris-train.csv",tableViewData);
        });
        buttonDataTest.setOnAction(evt->{
            viewDataSet("irisTest.csv",tableViewData);
        });

        buttonPredict.setOnAction(evt->{
            try {
                double sl=Double.parseDouble(textFieldSepalLength.getText());
                double sw=Double.parseDouble(textFieldSepalWidth.getText());
                double pl=Double.parseDouble(textFieldPetalLength.getText());
                double pw=Double.parseDouble(textFieldPetalWidth.getText());
                System.out.println("Prediction :");
                INDArray input= Nd4j.create(new double[][]{{sl,sw,pl,pw}});
                INDArray ouput=model.output(input);
                textAreaConsole.appendText(ouput.toString());
                String labelOutput=labels[Nd4j.argMax(ouput).getInt(0)];
                labelPrection.setText(labelOutput);
                imageViewPrediction.setImage(new Image(new FileInputStream(new ClassPathResource("images/"+labelOutput+".png").getFile())));
            } catch (IOException e) {
                e.printStackTrace();
            }

        });

    }

    private void viewDataSet(String fileName, TableView tableView) {
        try {
            File file=new ClassPathResource(fileName).getFile();
            RecordReader recordReader=new CSVRecordReader();
            recordReader.initialize(new FileSplit(file));
            DataSetIterator dataSetIterator=
                    new RecordReaderDataSetIterator(recordReader,batchSize,classIndex,outputSize);
           tableView.getItems().clear();

        while (dataSetIterator.hasNext()){
            DataSet dataSet=dataSetIterator.next();
            INDArray features=dataSet.getFeatures();
            INDArray targets=dataSet.getLabels();
            for (int k = 0; k <batchSize ; k++) {
                INDArray batchFeature=features.getRow(k);
                INDArray batchLabel=targets.getRow(k);
                Map<String,Object> data=new HashMap<>();
                double[] row=batchFeature.toDoubleVector();
                INDArray rowLabels=batchLabel.getRow(0);
                for (int j = 0; j <row.length ; j++) {
                    data.put(featuresLabels[j],row[j]);
                }
                data.put("Label",labels[rowLabels.argMax().getInt(0)]);
                tableView.getItems().add(data);
            }


        }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
