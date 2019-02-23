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
 * This topology counts the words from sentences emmited from a random sentence spout.
 */
public class TopWordFinderTopologyPartA {

  public static void main(String[] args) throws Exception {

    TopologyBuilder builder = new TopologyBuilder();

    builder.setSpout("spout", new RandomSentenceSpout(), 5);
    builder.setBolt("split", new SplitSentenceBolt(), 8).shuffleGrouping("spout");
    builder.setBolt("count", new WordCountBolt(), 12).fieldsGrouping("split", new Fields("word"));

    Config config = new Config();
    config.setDebug(true);

    if (args != null && args.length > 0) {
      config.setNumWorkers(3);
      StormSubmitter.submitTopology(args[0], config, builder.createTopology());
    }
    else {
      config.setMaxTaskParallelism(3);

      LocalCluster cluster = new LocalCluster();
      cluster.submitTopology("word-count", config, builder.createTopology());

      //wait for 60 seconds and then kill the topology
      Thread.sleep(60 * 1000);

      cluster.shutdown();
    }
  }
}
