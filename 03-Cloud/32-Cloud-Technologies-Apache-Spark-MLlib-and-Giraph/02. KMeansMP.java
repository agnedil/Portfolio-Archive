// CCA UIUC
// Implementation of KMeans clustering on a cars dataset 

import java.util.regex.Pattern;
import scala.Tuple2;									// a Tuple class from Scala (container for 2 - 22 elements; 2 in our case)

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;						// key-value pair based on RDD class (see below)
import org.apache.spark.api.java.JavaRDD;						// J. class for Spark's Resilient Distributed Dataset (inmutable collect.of objects)
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;					// Set of f(x) in Spark's Java API, passed to various Java API methods for Spark
import org.apache.spark.api.java.function.PairFunction;					// returns key-value pairs (Tuple2<K, V>), can be used to construct PairRDDs
import org.apache.spark.api.java.function.VoidFunction;					// a function with no return value

import org.apache.spark.mllib.clustering.KMeans;                                        // mllib - Spark machine learning library
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;						// base clsass -- Vectors, local class - Vector (dense and sparse)


public final class KMeansMP {
	//

    private static class ParsePoint implements Function<String, Vector> {		// parsing vector points from input data RDD
        private static final Pattern SPACE = Pattern.compile(",");			// split by comma

        public Vector call(String line) {						// class call f(x)
            String[] token = SPACE.split(line);						// split line into an array of Strings
            double[] point = new double[token.length - 1];				// array of doubles; "length-1" because token[0] is title (car name)
            for (int i = 1; i < token.length; ++i) {					// populate array with parsed doubles
                point[i - 1] = Double.parseDouble(token[i]);				// convert strings representing numbers to double
            }
            return Vectors.dense(point);
        }
    }

    private static class ParseTitle implements Function<String, String> {		// get title (car name) from the input data RDD of Strings
        private static final Pattern SPACE = Pattern.compile(",");

        public String call(String line) {
            String[] token = SPACE.split(line);
            return token[0];
        }
    }    

    private static class ClusterCars implements PairFunction<Tuple2<String, Vector>, Integer, String> {		// f(x) that predicts using a trained model
        private KMeansModel model;

        public ClusterCars(KMeansModel model) {
            this.model = model;
        }

        public Tuple2<Integer, String> call(Tuple2<String, Vector> args) {
            String title = args._1();							// access the first component of the arg passed to f(x) (of type Tuple2)
            Vector point = args._2();							// access the second such component
            int cluster = model.predict(point);						// kmeans prediction for this vector
            return new Tuple2<>(cluster, title);					// return as Tuple2
        }
    }

    public static void main(String[] args) {
        if (args.length < 2) {								// check for correct number of files indicated in the program call
            System.err.println(
                    "Usage: KMeansMP <input_file> <results>");
            System.exit(1);
        }
        String inputFile = args[0];
        String results_path = args[1];
        JavaPairRDD<Integer, Iterable<String>> results;					// key-value pair used to store model prediction results (below)
        int k = 4;									// kmeans parameters
        int iterations = 100;
        int runs = 1;
        long seed = 12345;
		
        SparkConf sparkConf = new SparkConf().setAppName("KMeans MP");			// set Spark configuration and context
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // process input data
        JavaRDD<String> lines = sc.textFile(inputFile);					// read text file from HDFS or any Hadoop file system, return RDD of Strings
        JavaRDD<Vector> points = lines.map(new ParsePoint());				// map the returned Strings and vector points
        JavaRDD<String> titles = lines.map(new ParseTitle());				// map the returned Strings and titles (car names)
        
	KMeansModel model = KMeans.train(points.rdd(), k, iterations, runs, KMeans.RANDOM(), seed);		// train model
        results = titles.zip(points).mapToPair(new ClusterCars(model)).groupByKey();				// predict using a ClusterCars call

        results.saveAsTextFile(results_path);									// save results

        sc.stop();												// stop Spark context
    }
}
