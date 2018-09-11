// CCA 2018 UIUC
// MP5
// using Giraph and Hadoop to find the shortest path in a graph

import java.io.IOException;
import org.apache.hadoop.io.IntWritable;				// Hadoop variant of integer, implements its own methods, less bulky for big data than integer
import org.apache.hadoop.io.NullWritable;				// Hadoop variant of null, implements its own methods, less bulky for big data than null
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.conf.LongConfOption;				// Long configuration option
import org.apache.giraph.edge.Edge;
import org.apache.giraph.graph.Vertex;


// Compute shortest paths from a given source
public class ShortestPathsComputation extends BasicComputation<
    IntWritable, IntWritable, NullWritable, IntWritable> {

  // Shortest paths id
  public static final LongConfOption SOURCE_ID =			// new var of type LongConfOption
      new LongConfOption("SimpleShortestPathsVertex.sourceId", 1,
          "The shortest paths id");

  // Is this vertex the source id? @params vertex Vertex, returns True if the source id
  private boolean isSource(Vertex<IntWritable, ?, ?> vertex) {
    return vertex.getId().get() == SOURCE_ID.get(getConf());		// statement with a boolean outcome
  }

  // compute() propagates info to all neighbors about the shortest path from a given vertex 
  // @params vertex Vertex, messages Iterator of messages from previous superstep, throws IOException
  @Override
  public void compute(
        Vertex<IntWritable, IntWritable, NullWritable> vertex,
        Iterable<IntWritable> messages) throws IOException {
      
        if(getSuperstep() == 0){
            vertex.setValue(new IntWritable(Integer.MAX_VALUE));
        }
        
        int minDist = isSource(vertex) ? 0 : Integer.MAX_VALUE;
        for (IntWritable message : messages){
            minDist = Math.min(minDist, message.get());						// find minimum distance
        }
        
        if (minDist < vertex.getValue().get()){							// if minimum distance is smaller than the vertex's value
            vertex.setValue(new IntWritable(minDist));
            
            for (Edge<IntWritable, NullWritable> edge : vertex.getEdges()){     
                sendMessage(edge.getTargetVertexId(), new IntWritable(minDist + 1));		// propagate new distance to all neighbor vertices
            }
        }
        vertex.voteToHalt();									// compute() will not be called for this vertex unless message is sent to it
  }
}
