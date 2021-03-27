package nndescent;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;

public class NNDescentTest {
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.setInt("dimVector",25);
        conf.setInt("k", 5);
        conf.setInt("sampleSize", 5);
        conf.setInt("numVectors", 1183514);
        String inputPath = "./data/glove_25d_train.txt";
        String outputPath = "./data/25d_nnd";

        ToolRunner.run(conf, new NNDescent(), new String[]{inputPath, outputPath});
    }
}
