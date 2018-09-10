import java.io.IOException;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.File;
import java.util.Scanner;

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

import org.apache.hadoop.hbase.util.Bytes;

public class TablePartC{

   public static void main(String[] args) throws IOException {
        // Instantiating Configuration class	
        Configuration config = HBaseConfiguration.create();

        // Instantiating HTable class	
        HTable hTable = new HTable(config, "powers");

        // connecting Scanner object with file
        Scanner inputStream = new Scanner(new File("input.csv"));
                
        // row number
        int j = 1;

        // loop - streaming lines from csv file, splitting, adding to table
        while (inputStream.hasNextLine()) {

            String line = inputStream.nextLine();
            line = line.trim();
            String[] records = line.split(",");
            
            // check if record is good; TODO: come up with a better way
            if (records.length != 6){
                System.out.println("Bad record: " + line);
            } //if            
           
            // create string "row#"           
            String rowNum = "row" + j;            
           
            // Instantiating Put class; accepts row name	
            Put p = new Put(Bytes.toBytes(rowNum));

            // adding values using add() method
            // accepts column family name, qualifier/row name, value	
            p.add(Bytes.toBytes("personal"), Bytes.toBytes("hero"),
                    Bytes.toBytes(records[1]));

            p.add(Bytes.toBytes("personal"), Bytes.toBytes("power"),
                    Bytes.toBytes(records[2]));

            p.add(Bytes.toBytes("professional"), Bytes.toBytes("name"),
                    Bytes.toBytes(records[3]));

            p.add(Bytes.toBytes("professional"), Bytes.toBytes("xp"),
                    Bytes.toBytes(records[4]));

            p.add(Bytes.toBytes("custom"), Bytes.toBytes("color"),
                    Bytes.toBytes(records[5]));

            // Saving put Instance to the HTable	
            hTable.put(p);
            //System.out.println("data	inserted");
            
            // increase row number by one
            j++;

       }  //while
       
       // closing HTable and Scanner object
       hTable.close();
       inputStream.close();
       
   }
}