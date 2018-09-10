
import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.StormSubmitter;
import org.apache.storm.topology.BasicOutputCollector;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.TopologyBuilder;
import org.apache.storm.topology.base.BaseBasicBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;

/**
 * This topology reads a file, splits the senteces into words, normalizes the words such that all words are
 * lower case and common words are removed, and then count the number of words.
 */
public class TopWordFinderTopologyPartC {

  public static void main(String[] args) throws Exception {


    TopologyBuilder builder = new TopologyBuilder();

    Config config = new Config();
    config.setDebug(true);

    /* TODO:
    Wire up the topology using setBolt(name,…) and setSpout(name,…),
    and the following names:
    FileReaderSpout -> "spout", SplitSentenceBolt -> "split"
    NormalizerBolt -> "normalize", WordCountBolt -> "count" */
        
    builder.setSpout("spout", new FileReaderSpout(), 1);
    builder.setBolt("split", new SplitSentenceBolt(), 8).shuffleGrouping("spout");
    builder.setBolt("normalize", new NormalizerBolt(), 10).shuffleGrouping("split");                 // use new Fileds?
    builder.setBolt("count", new WordCountBolt(), 12).fieldsGrouping("normalize", new Fields("word"));
    
    config.setMaxTaskParallelism(3);

    LocalCluster cluster = new LocalCluster();
    cluster.submitTopology("word-count", config, builder.createTopology());

    //wait for 2 minutes then kill the job
    Thread.sleep(2 * 60 * 1000);

    cluster.shutdown();
  }
}
