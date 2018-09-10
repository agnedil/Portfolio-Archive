// CCA 2018 UIUC
// MP5
// hints from: https://apache.googlesource.com/giraph/+/trunk/giraph-examples/src/main/java/org/apache/giraph/examples/ConnectedComponentsComputation.java
// using Giraph and Hadoop to identify connected componenets in a graph

import java.io.IOException;
import org.apache.hadoop.io.IntWritable;		// Hadoop variant of Integer for Hadoop serialization. Integer uses Java Serialization - too bulky and costly in Hadoop's handling of hyge amounts of data
// 3 main methods: Comparable, Writable and WritableComparable (all are more efficient for Hadoop
import org.apache.hadoop.io.NullWritable;		// Hadoop variant of null

import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.graph.Vertex;

/**
 * ConnectedComponentsComputation class implements the connected component algorithm that identifies
 * connected components and assigns each vertex its "component
 * identifier" (the smallest vertex id in the component).
 */

public class ConnectedComponentsComputation extends
    BasicComputation<IntWritable, IntWritable, NullWritable, IntWritable> {
  /**
   * Compute() propagates smallest vertex id to all neighbors; halts
   * and reactivates only if a smaller id is received.
   *
   * @params vertex Vertex, messages Iterator of messages from the previous superstep.
   * @throws IOException
   */
  @Override
  public void compute(
        Vertex<IntWritable, IntWritable, NullWritable> vertex,
        Iterable<IntWritable> messages) throws IOException {
     
        int my_id = vertex.getValue().get();
        
        if (getSuperstep() == 0){
            // Look at the neighbors
            for (Edge<IntWritable, NullWritable> edge : vertex.getEdges()){			// iterate over edges
                IntWritable neighbor = edge.getTargetVertexId();				// get neighbor vetices for edges found
                my_id = Math.min(my_id, neighbor.get());					// see if their id is smaller
            }
            
            // Send value only if it is not my_id 
            if (my_id != vertex.getValue().get()){
                vertex.setValue(new IntWritable(my_id));
                
                for (Edge<IntWritable, NullWritable> edge : vertex.getEdges()){
                    IntWritable neighbor = edge.getTargetVertexId();
                    if (neighbor.get() > my_id){
                        sendMessage(neighbor, vertex.getValue());				// send message if neighbor is larger this vertex's id
                    }
                }
            }
        }
        else{
            // Did we get a smaller id?
            for (IntWritable msg : messages){
                my_id = Math.min(my_id, msg.get());						// see if the id received is smaller
            }

            // Send new id to neighbors
            if (my_id != vertex.getValue().get()){
                vertex.setValue(new IntWritable(my_id));
                sendMessageToAllEdges(vertex, vertex.getValue());
            }
        }
        vertex.voteToHalt();									// compute() will not be called for this vertex unless message is sent to it
    }
}

