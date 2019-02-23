// CCA UIUC
// Implementation of RandomForest classification on a cars dataset 


import java.util.HashMap;								// basic implementation of Map; stores (Key,Value) pairs; need to know key to
import java.util.regex.Pattern;								// access value; uses Hashing to store longer strings as shorter strings

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;						// J. class for Spark's Resilient Distributed Dataset (inmutable collect.of objects)
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;					// Set of f(x) in Spark's Java API, passed to various Java API methods for Spark
import org.apache.spark.mllib.classification.SVMModel;					// Does Random Forest use support vector machine
import org.apache.spark.mllib.classification.SVMWithSGD;				// and stochastic gradient descent? Not sure why these 2 imports
import org.apache.spark.mllib.linalg.Vector;						// mllib - Spark machine learning library
import org.apache.spark.mllib.linalg.Vectors;						// base clsass -- Vecotrs, local class - Vector (dense and sparse)
import org.apache.spark.mllib.regression.LabeledPoint;					// class that represents the features and labels of a data point
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.tree.RandomForest;


public final class RandomForestMP {
	//TODO

    private static class DataToPoint implements Function<String, LabeledPoint> {			// parsing training set (keep label = last item in the line)
        private static final Pattern SPACE = Pattern.compile(",");					// splitting at ","

        public LabeledPoint call(String line) throws Exception {
            String[] token = SPACE.split(line);
            double label = Double.parseDouble(token[token.length - 1]);					// get label
            double[] point = new double[token.length - 1];						// initialize array for datapoints
            for (int i = 0; i < token.length - 1; ++i) {						// get datapoints
                point[i] = Double.parseDouble(token[i]);
            }
            return new LabeledPoint(label, Vectors.dense(point));
        }
    }

    private static class DataToVector implements Function<String, Vector> {				// parsing training set (do not keep label = last item in the line)
        private static final Pattern SPACE = Pattern.compile(",");

        public Vector call(String line) throws Exception {
            String[] token = SPACE.split(line);
            double[] point = new double[token.length - 1];
            for (int i = 0; i < token.length - 1; ++i) {						// get datapoints in the array without labels
                point[i] = Double.parseDouble(token[i]);
            }
            return Vectors.dense(point);
        }
    }

    public static void main(String[] args) {
        if (args.length < 3) {										// see if correct files were specified in program call
            System.err.println(
                    "Usage: RandomForestMP <training_data> <test_data> <results>");
            System.exit(1);
        }
        String training_data_path = args[0];								// meaning of filenames in program call
        String test_data_path = args[1];
        String results_path = args[2];

        SparkConf sparkConf = new SparkConf().setAppName("RandomForestMP");				// set Spark configuration and context
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
	final RandomForestModel model;

	JavaRDD<LabeledPoint> train = sc.textFile(training_data_path).map(new DataToPoint());		// parse train and test sets using the above 2 different f(x)
        JavaRDD<Vector> test = sc.textFile(test_data_path).map(new DataToVector());

        Integer numClasses = 2;										// model parameters
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 3;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;

	// train model
	model = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);


        JavaRDD<LabeledPoint> results = test.map(new Function<Vector, LabeledPoint>() {
            public LabeledPoint call(Vector points) {
                return new LabeledPoint(model.predict(points), points);					// predict step
            }
        });

        results.saveAsTextFile(results_path);								// save results

        sc.stop();											// stop Spark context
    }

}
