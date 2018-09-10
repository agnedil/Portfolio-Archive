import java.io.IOException;

import org.apache.hadoop.conf.Configuration;

import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;

import org.apache.hadoop.hbase.TableName;

import org.apache.hadoop.hbase.client.HBaseAdmin;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.ResultScanner;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.client.Get;


import org.apache.hadoop.hbase.util.Bytes;

public class TablePartD{

    
   public static void main(String[] args) throws IOException {
        // Instantiating Configuration class	
        Configuration config = HBaseConfiguration.create();

        // Instantiating HTable class	
        HTable table = new HTable(config, "powers");

        // Instantiating Get class for row 1	
        Get g1 = new Get(Bytes.toBytes("row1"));

        // Reading the data	
        Result result = table.get(g1);

        // Reading values from Result class object	
        byte[] value1 = result.getValue(Bytes.toBytes("personal"), Bytes.toBytes("hero"));
        byte[] value2 = result.getValue(Bytes.toBytes("personal"), Bytes.toBytes("power"));
        byte[] value3 = result.getValue(Bytes.toBytes("professional"), Bytes.toBytes("name"));
        byte[] value4 = result.getValue(Bytes.toBytes("professional"), Bytes.toBytes("xp"));
        byte[] value5 = result.getValue(Bytes.toBytes("custom"), Bytes.toBytes("color"));

        // Printing the values	
        String hero = Bytes.toString(value1);
        String power = Bytes.toString(value2);
        String name = Bytes.toString(value3);
        String xp = Bytes.toString(value4);
        String color = Bytes.toString(value5);

        System.out.println("hero: " + hero + ", power: " + power + ", name: " + name + ", xp: " + xp + ", color: " + color);
                
        // Instantiating Get class for row 19	
        Get g19 = new Get(Bytes.toBytes("row19"));

        // Reading the data	
        Result result2 = table.get(g19);

        // Reading values from Result class object	
        byte[] value6 = result2.getValue(Bytes.toBytes("personal"), Bytes.toBytes("hero"));        
        byte[] value7 = result2.getValue(Bytes.toBytes("custom"), Bytes.toBytes("color"));

        // Printing the values	
        String hero19 = Bytes.toString(value6);        
        String color19 = Bytes.toString(value7);

        System.out.println("hero: " + hero19 + ", color: " + color19);
        
        //repeated print for row 1, but in a different sequence; data already in memory
        System.out.println("hero: " + hero + ", name: " + name + ", color: " + color);
                
    } // main
} // class