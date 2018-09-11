// some from here https://github.com/hackreduce/storm-hackathon/blob/master/src/main/java/org/hackreduce/storm/LineSpout.java


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Map;
import java.io.File;
import java.io.InputStreamReader;
import java.util.logging.Level;
import java.util.logging.Logger;

//change backtype to org.apache
import org.apache.storm.spout.SpoutOutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.IRichSpout;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Values;
import org.apache.storm.utils.Utils;

public class FileReaderSpout implements IRichSpout {  
  private SpoutOutputCollector _collector;
  private TopologyContext context;  
  private BufferedReader reader;


  @Override
  public void open(Map config, TopologyContext context,
                   SpoutOutputCollector collector) {
    // TODO: initialize file reader
        
    _collector = collector;
    try {
        reader = new BufferedReader(new FileReader("data.txt"));
    } catch (IOException e) {	
        e.printStackTrace();
    }    
  }

  @Override
  public void nextTuple() {
    // TODO: 1) read next line, emit tuple for it 2) sleep when file is entirely read
    
    // Utils.sleep(100);        
    String line;
    
    try {
        while((line = reader.readLine())!=null) {
            _collector.emit(new Values(line));
        } } catch (IOException e) {
        e.printStackTrace();
    }
    Utils.sleep(100);     
  }

  @Override
  public void declareOutputFields(OutputFieldsDeclarer declarer) {

    declarer.declare(new Fields("line"));

  }

  @Override
  public void close() {
    // TODO: close the file
    try {
      reader.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override
  public void activate() {
  }

  @Override
  public void deactivate() {
  }

  @Override
  public void ack(Object msgId) {
  }

  @Override
  public void fail(Object msgId) {
  }

  @Override
  public Map<String, Object> getComponentConfiguration() {
    return null;
  }
}
